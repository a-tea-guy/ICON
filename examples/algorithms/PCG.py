from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import SupConLoss
from models.initializer import initialize_model
from models.pcg_network import PCGNetwork
from algorithms.single_model_algorithm import SingleModelAlgorithm
from wilds.common.utils import split_into_groups
from configs.supported import process_pseudolabels_functions
import torch.autograd as autograd
from models.initializer import initialize_model
from optimizer import initialize_optimizer_with_model_params, PCGrad
from wilds.common.metrics.metric import ElementwiseMetric, MultiTaskMetric
from optimizer import initialize_optimizer
from utils import move_to, Visualizer, detach_and_clone, load
from tqdm import tqdm
import os
import numpy as np
from utils import move_to, collate_list, concat_input
from train import infer_predictions, learn_pcg_unlabel_head


class PCGradPretrain(SingleModelAlgorithm):

    def __init__(self, config, d_out, unlabel_head, grouper, loss, unlabeled_loss, metric, n_train_steps):
        """
        Algorithm-specific arguments (in config):
            - irm_lambda
            - irm_penalty_anneal_iters
        """
        # check config
        # assert config.train_loader == 'group'
        # assert config.uniform_over_groups
        # assert config.distinct_groups
        # Initialize model
        featurizer, classifier = initialize_model(
            config, d_out=d_out, is_featurizer=True
        )
        self.feat_dim = featurizer.d_out
        self.num_classes = d_out
        self.super_contra = config.pcg_super_contra
        self.use_mlp_head = config.use_mlp_head
        self.use_mask = config.use_mask
        model = PCGNetwork(
            featurizer, classifier, unlabel_head, dropout=config.pcg_dropout,
            sup_contra=self.super_contra, use_mlp_head=config.use_mlp_head,
            use_mask=self.use_mask
        )
        self.optimizer = initialize_optimizer_with_model_params(config, model.parameters())
        if config.pc_grad:
            self.optimizer = PCGrad(self.optimizer, strength=config.grad_lambda)
        
        # initialize the module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )

        # set EqInv-specific variables
        self.unlabeled_loss = unlabeled_loss
        self.iteration = 0
        self.cluster_steps = 0
        self.config = config

        if self.super_contra:
            self.sup_contra_loss = SupConLoss(temperature=config.supcontra_temperature)

        # Plot losses
        self.visualizer = Visualizer(config.tensorboard_dir, config.exp_name)
        assert isinstance(self.loss, ElementwiseMetric) or isinstance(self.loss, MultiTaskMetric)

    def update_pseudo_labels(self, featurizer, classifier, by_cluster):
        if by_cluster:
            if classifier is None:
                print("Learning unlabel classifier")
                classifier, steps = learn_pcg_unlabel_head(
                    featurizer, self.feat_dim, self.num_classes,
                    self.two_view_loader, self.config, u_head=None,
                    cluster_steps=self.cluster_steps
                )
                self.cluster_steps = steps
        model = nn.Sequential(featurizer, classifier)
        pseudo_labels = infer_predictions(
            model, self.seq_loader, self.config, two_views=True
        )
        pseudo_labels = move_to(pseudo_labels, torch.device("cpu"))
        self.unlabel_dataset.pseudolabels = pseudo_labels
        featurizer.train()  # This function is only called during train
        classifier.train()  # This function is only called during train
        return classifier

    def get_pseudo_labels_udpate_parameters(self):
        pcg_uhead_path = self.config.pcg_uhead_path
        pcg_ps_update_freq = self.config.pcg_ps_update_freq
        pcg_pretrain_epochs = self.config.pcg_pretrain_epochs
        start_of_update_epoch = (pcg_ps_update_freq > 0) \
                        and (self.current_epoch % pcg_ps_update_freq == 0)
        if not (start_of_update_epoch and self.is_training):
            return None, None, False
        if hasattr(self.model, 'module'):
            model_without_module = self.model.module
        else:
            model_without_module = self.model
        if self.current_epoch < pcg_pretrain_epochs:
            if self.current_epoch != 0:
                return None, None, True
            # Use unlabel clustering only at the begining
            featurizer = model_without_module.featurizer
            if pcg_uhead_path != '':
                classifier = torch.nn.Linear(
                    self.feat_dim, self.num_classes
                ).to(self.config.device)
                classifier.load_state_dict(torch.load(pcg_uhead_path))
                print("Loaded unlabel classifier from %s" % pcg_uhead_path)
            else:
                classifier = None
            by_cluster = True
        else:
            # Use source-trained head
            featurizer = model_without_module.featurizer
            classifier = model_without_module.classifier
            by_cluster = False
        return featurizer, classifier, by_cluster

    def start_of_epoch_setup(self, epoch):
        featurizer, classifier, by_cluster = self.get_pseudo_labels_udpate_parameters()
        if featurizer is not None:
            updated_classifier = self.update_pseudo_labels(featurizer, classifier, by_cluster)
            if hasattr(self.model, 'module'):
                model_without_module = self.model.module
            else:
                model_without_module = self.model
            if by_cluster:
                model_without_module.u_head = updated_classifier
            else:
                model_without_module.u_head = None    # Stop clustering and use l_head only

    def process_batch(self, batch, unlabeled_batch=None):
        use_two_views = self.is_training and self.super_contra
        if use_two_views:
            (x_weak, x_strong), y_true, metadata = batch
            x = concat_input(x_weak, x_strong)
            n_lab = len(metadata) * 2
        else:
            x, y_true, metadata = batch
            n_lab = len(metadata)
        x = move_to(x, self.device)
        y_true = move_to(y_true, self.device)
        g = move_to(self.grouper.metadata_to_group(metadata), self.device)
        # package the results
        results = {"g": g, "y_true": y_true, "metadata": metadata}

        # Unlabeled examples with pseudolabels
        if unlabeled_batch is not None:
            (x_unlab_weak, x_unlab_strong), y_pseudo, metadata_unlab = unlabeled_batch
            if use_two_views:
                x_unlab = concat_input(x_unlab_weak, x_unlab_strong)
            else:
                x_unlab = x_unlab_strong
            x_unlab = move_to(x_unlab, self.device)
            g_unlab = move_to(self.grouper.metadata_to_group(metadata_unlab), self.device)
            y_pseudo = move_to(y_pseudo, self.device)
            results["unlabeled_metadata"] = metadata_unlab
            results["unlabeled_y_pseudo"] = y_pseudo
            results["unlabeled_g"] = g_unlab
            x_cat = concat_input(x, x_unlab)
            # y_cat = collate_list([y_true, y_pseudo]) if self.model.needs_y else None
            features, outputs_l, outputs_u, l_view, u_view = self.model(x_cat)

            # get predictions
            if not use_two_views:
                results["y_pred"] = outputs_l[:n_lab]
                results["unlabeled_y_pred"] = outputs_u[n_lab:]
            else:
                results["y_pred"] = torch.chunk(outputs_l[:n_lab], 2)[1]    # strong aug
                results["unlabeled_y_pred"] = torch.chunk(outputs_u[n_lab:], 2)[1]    # strong aug

            # get sup contrastive views
            if use_two_views:
                if self.use_mlp_head:
                    results["l_views"] = torch.stack(torch.chunk(l_view[:n_lab], 2)).swapaxes(0, 1) # bs*2*feat_dim
                    results["u_views"] = torch.stack(torch.chunk(u_view[n_lab:], 2)).swapaxes(0, 1)
                else:
                    results["l_views"] = torch.stack(torch.chunk(features[:n_lab], 2)).swapaxes(0, 1)
                    results["u_views"] = torch.stack(torch.chunk(features[n_lab:], 2)).swapaxes(0, 1)
                results["l_views"] = F.normalize(results["l_views"], dim=-1)
                results["u_views"] = F.normalize(results["u_views"], dim=-1)
        else:
            features, outputs_l, _, l_view, _ = self.model(x)
            if use_two_views:
                results["y_pred"] = torch.chunk(outputs_l, 2)[1]    # strong aug
            else:
                results["y_pred"] = outputs_l
            if use_two_views:
                # construct two views
                if self.use_mlp_head:
                    results["l_views"] = torch.stack(torch.chunk(l_view, 2)).swapaxes(0, 1)
                else:
                    results["l_views"] = torch.stack(torch.chunk(features, 2)).swapaxes(0, 1)
                results["l_views"] = F.normalize(results["l_views"], dim=-1)
        return results

    def objective(self, results):
        if self.is_training:
            self.iteration += 1
        use_two_views = self.is_training and self.super_contra

        # Labeled loss
        classification_loss = self.loss.compute(
            results["y_pred"], results["y_true"], return_dict=False
        )
        # Pseudolabel loss
        if "unlabeled_y_pseudo" in results:
            consistency_loss = self.unlabeled_loss.compute(
                results["unlabeled_y_pred"],
                results["unlabeled_y_pseudo"],
                return_dict=False,
            )
        else:
            consistency_loss = 0.0
        # Supervised Contrastive loss
        if use_two_views:
            # Labeled supervised contrastive
            label_contra_loss = self.sup_contra_loss(
                results["l_views"], results["y_true"]
            )
            # Unlabeled supervised contrastive
            if "unlabeled_y_pseudo" in results:
                unlabel_contra_loss = self.sup_contra_loss(
                    results["u_views"], results["unlabeled_y_pseudo"]
                )
            else:
                unlabel_contra_loss = 0.0
            l_bs = len(results["y_true"])
            u_bs = len(results["unlabeled_y_pseudo"])
            contra_loss = (l_bs * label_contra_loss + u_bs * unlabel_contra_loss) / (l_bs + u_bs)
        else:
            contra_loss = 0.0

        total_loss = classification_loss\
                    + self.config.consistency_lambda * consistency_loss\
                    + self.config.supcontra_lambda * contra_loss
        # Plot on tensorboard
        if self.is_training:
            visualize_items = {
                "ERM loss": classification_loss,
                "Consistency loss": consistency_loss,
                "Supervised contrastive loss": contra_loss
            }
            self.visualizer.plot_items(self.iteration, visualize_items)

        if (not self.config.pc_grad) or (not self.is_training):
            return total_loss
        else:
            losses = [
                classification_loss,
                self.config.consistency_lambda * consistency_loss,
                self.config.supcontra_lambda * contra_loss
            ]
            return total_loss, losses