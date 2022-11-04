from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import SupConLoss
from models.initializer import initialize_model
from examples.models.icon_network import RegDNetwork
from algorithms.single_model_algorithm import SingleModelAlgorithm
from wilds.common.utils import split_into_groups
from configs.supported import process_pseudolabels_functions
import torch.autograd as autograd
from models.initializer import initialize_model
from optimizer import initialize_optimizer_with_model_params
from wilds.common.metrics.metric import ElementwiseMetric, MultiTaskMetric
from optimizer import initialize_optimizer
from utils import move_to, Visualizer, detach_and_clone, load, collect_feature, reduce_dimension
from tqdm import tqdm
import os
import numpy as np
from utils import move_to, collate_list, concat_input
from train import infer_predictions, learn_pcg_unlabel_head
from examples.icon_losses import ClusterLoss, DisentanglementLoss, Normalize, BCE, get_ulb_sim_matrix, PairEnum, TsallisEntropy, calc_entropy, calc_diversity, compute_ulb_sim_matrix
from utils import get_model_prefix, load


class RegDet(SingleModelAlgorithm):
    def __init__(self, config, d_out, grouper, loss, unlabeled_loss, metric, n_train_steps):
        """
        Algorithm-specific arguments (in config):
            - irm_lambda
            - irm_penalty_anneal_iters
        """

        model = initialize_model(config, d_out=d_out, is_featurizer=False)
        self.process_pseudolabels_function = process_pseudolabels_functions[
            config.process_pseudolabels_function
        ]
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        self.unlabeled_loss = unlabeled_loss
        self.d_out = d_out
        # additional logging
        self.logged_fields.append("classification_loss")
        self.logged_fields.append("consistency_loss")
        self.visualizer = Visualizer(config.log_dir)
        self.iteration = 0
        self.config = config

    def start_of_epoch_setup(self, epoch):
        args = self.config
        # update pseudo label if necessary
        is_update_epoch = (epoch - self.config.consistency_start_epoch) % self.config.regd_ps_freq == 0
        if is_update_epoch:
            assert self.config.teacher_model_path is not None
            print("Updating pseudo labels...")
            self.update_pseudo_labels(epoch)

        # Update loss weights
        self.w_cluster = args.w_cluster
        self.w_compat_irm = args.w_compat_irm if epoch >= args.compat_irm_start_epoch else 0.0
        self.w_erm = 1.0
        self.w_compat = args.w_compat if epoch >= args.compat_start_epoch else 0.0
        self.w_con = args.w_con if epoch >= args.consistency_start_epoch else 0.0

        # update epoch
        self.model.roi_heads.epoch = epoch

    def update_pseudo_labels_debug(self, epoch):
        teacher_outputs = torch.load("wheat.pt")
        teacher_outputs = move_to(teacher_outputs, torch.device("cpu"))
        self.unlabel_dataset.pseudolabels = teacher_outputs

    def update_pseudo_labels(self, epoch):
        config = self.config
        if epoch == self.config.consistency_start_epoch:
            # use pretrained teacher
            teacher_model = initialize_model(config, d_out=self.d_out, is_featurizer=False).to(self.device)
            load(teacher_model, config.teacher_model_path, device=config.device)
            self.logger.write("\nLoading from %s...\n" % config.teacher_model_path)
        else:
            # load current best
            teacher_model = initialize_model(config, d_out=self.d_out, is_featurizer=False).to(self.device)
            prefix = self.prefix
            best_model_dir = prefix + 'epoch:best_model.pth'
            self.logger.write("\nLoading teacher from %s...\n" % best_model_dir)
            load(teacher_model, best_model_dir, config.device)
            self.logger.write("\nLoading model from %s...\n" % best_model_dir)
            load(self.model, best_model_dir, config.device)

        # infer labels
        teacher_outputs = infer_predictions(teacher_model, self.seq_loader, config)
        teacher_outputs = move_to(teacher_outputs, torch.device("cpu"))
        self.unlabel_dataset.pseudolabels = teacher_outputs
        teacher_model = teacher_model.to(torch.device("cpu"))
        del teacher_model

    def process_batch(self, batch, unlabeled_batch=None):
        # Labeled examples
        x, y_true, metadata = batch
        n_lab = len(metadata)
        x = move_to(x, self.device)
        y_true = move_to(y_true, self.device)
        g = move_to(self.grouper.metadata_to_group(metadata), self.device)
        # package the results
        results = {"g": g, "y_true": y_true, "metadata": metadata}

        # Unlabeled examples with pseudolabels
        if unlabeled_batch is not None:
            x_unlab, y_pseudo, metadata_unlab = unlabeled_batch
            x_unlab = move_to(x_unlab, self.device)
            g_unlab = move_to(self.grouper.metadata_to_group(metadata_unlab), self.device)
            y_pseudo = move_to(y_pseudo, self.device)
            results["unlabeled_metadata"] = metadata_unlab
            results["unlabeled_y_pseudo"] = y_pseudo
            results["unlabeled_g"] = g_unlab

            x_cat = concat_input(x, x_unlab)
            y_cat = collate_list([y_true, y_pseudo]) if self.model.needs_y else None
            outputs = self.get_model_output(x_cat, y_cat)
            results["y_pred"] = outputs[0][:n_lab]
            results["unlabeled_y_pred"] = outputs[0][n_lab:]
            results["icon"] = outputs[1]
        else:
            results["y_pred"] = self.get_model_output(x, y_true)

        return results

    def objective(self, results):
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
            icon = results["icon"]
            cluster_loss = icon["bce_loss"]
            compat_loss = icon["compat_loss"]
            compat_irm_loss = icon["compat_irm_loss"]
            cluster_entropy = icon["cluster_entropy"]
            batch_diversity = icon["batch_diversity"]
            irm_low_p_t = icon["irm_low_p_t"]
            irm_low_p_s = icon["irm_low_p_s"]
            back_cluster = icon["back_cluster"]
        else:
            consistency_loss = 0
            cluster_loss = 0
            compat_loss = 0
            compat_irm_loss = 0
            cluster_entropy = 0
            batch_diversity = 0
            irm_low_p_t = 0
            irm_low_p_s = 0
            back_cluster = 0

        # Add to results for additional logging
        self.save_metric_for_logging(
            results, "classification_loss", classification_loss
        )
        self.save_metric_for_logging(results, "consistency_loss", consistency_loss)

        if self.is_training:
            self.iteration += 1
            self.visualizer.plot_items(self.iteration, {
                "ERM loss": classification_loss,
                "Consistency loss": consistency_loss,
                "Cluster loss": cluster_loss,
                "Compat loss": compat_loss,
                "Compat IRM loss": compat_irm_loss,
                "Cluster entropy": cluster_entropy,
                "Batch diversity": batch_diversity,
                "IRM low percent target": irm_low_p_t,
                "Low percent source": irm_low_p_s,
                "Back cluster": back_cluster,
                "Compat low t": icon["low_t"],
                "Compat low p": icon["low_p"],
                "Compat high t": icon["high_t"],
                "Compat high p": icon["high_p"]
            })
        loss = classification_loss + self.w_con * consistency_loss\
                + self.w_compat_irm * compat_irm_loss\
                + self.w_compat * compat_loss\
                + self.w_cluster * cluster_loss
        return loss