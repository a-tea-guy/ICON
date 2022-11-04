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


class RegDisent(SingleModelAlgorithm):
    def __init__(self, config, d_out, grouper, loss, unlabeled_loss, metric, n_train_steps):
        """
        Algorithm-specific arguments (in config):
            - irm_lambda
            - irm_penalty_anneal_iters
        """
        # Task specific parameters
        self.is_regression = (config.dataset == "poverty")
        self.is_classification = not self.is_regression
        # check config
        # assert config.train_loader == 'group'
        # assert config.uniform_over_groups
        # assert config.distinct_groups
        # Initialize model
        featurizer, classifier = initialize_model(
            config, d_out=d_out, is_featurizer=True
        )
        self.feat_dim = featurizer.d_out
        self.num_classes = d_out if self.is_classification else config.num_pseudo_classes
        model = RegDNetwork(
            featurizer, classifier,
            bottleneck_dim=config.bottleneck_dim,
            bottleneck_cluster=config.bottleneck_cluster,
            use_additional_u_head=(config.w_compat > 0.0),
            dropout=config.regd_dropout, num_classes=self.num_classes
        )
        self.optimizer = initialize_optimizer_with_model_params(config, model.parameters())
        self.process_pseudolabels_function = process_pseudolabels_functions[config.process_pseudolabels_function]
        self.confidence_threshold = config.self_training_threshold

        # initialize the module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        self.unlabeled_loss = unlabeled_loss
        self.l2norm = Normalize(2)
        self.bce = BCE()
        self.ts = TsallisEntropy(
            temperature=config.ts_temperature, alpha=config.ts_alpha
        )
        self.back_cluster = False
        self.compat_sim_ratio = config.compat_sim_ratio
        self.compat_diff_ratio = config.compat_diff_ratio

        self.iteration = 0
        self.config = config

        # initialize cluster and disentangle loss
        diff_threshold = config.bce_diff_threshold if self.is_regression else None
        self.cluster_loss = ClusterLoss(
            self.device, self.num_classes, config.bce_type,
            config.cosine_threshold, config.topk, diff_threshold
        )
        self.disentangle_loss = DisentanglementLoss(
            self.num_classes, temperature=config.supcon_temperature,
            use_target_as_positive=False,
            memory_length_per_class=500
        )
        # Plot losses
        self.visualizer = Visualizer(config.log_dir)
        assert isinstance(self.loss, ElementwiseMetric) or isinstance(self.loss, MultiTaskMetric)

    def start_of_epoch_setup(self, epoch):
        args = self.config
        # update pseudo label if necessary
        should_update_ps_labels = self.config.noisy_student\
                and epoch >= self.config.consistency_start_epoch and self.config.w_con != 0.0
        is_update_epoch = (epoch - self.config.consistency_start_epoch) % self.config.regd_ps_freq == 0
        if should_update_ps_labels and is_update_epoch:
            assert self.config.teacher_model_path is not None
            print("Updating pseudo labels...")
            self.update_pseudo_labels(epoch)
        
        # Dimension reduction
        do_reduce_dim =  (args.dim_reduction != 'none')
        if do_reduce_dim:
            self.get_reduced_feature()

        # Update loss weights
        self.w_cluster = args.w_cluster
        self.w_sup = args.w_supcon if epoch >= args.irm_start_epoch else 0.0
        self.w_transfer = args.w_transfer
        self.w_irm = args.w_irm if epoch >= args.irm_start_epoch else 0.0
        self.w_compat_irm = args.w_compat_irm if epoch >= args.compat_irm_start_epoch else 0.0
        self.w_erm = 1.0
        self.w_compat = args.w_compat if epoch >= args.compat_start_epoch else 0.0
        self.w_con = args.w_con if epoch >= args.consistency_start_epoch else 0.0
        self.back_cluster = args.back_cluster if epoch >= args.back_cluster_start_epoch else False
        self.do_reduce_dim = do_reduce_dim
        if epoch > args.compat_start_epoch:
            self.compat_sim_ratio = (1.0 - self.compat_sim_ratio) * args.compat_sim_ratio_delta\
                                        + self.compat_sim_ratio
            self.compat_diff_ratio = (1.0 - self.compat_diff_ratio) * args.compat_diff_ratio_delta\
                                        + self.compat_diff_ratio
        else:
            self.compat_sim_ratio = args.compat_sim_ratio
            self.compat_diff_ratio = args.compat_diff_ratio

    def get_reduced_feature(self,):
        config = self.config
        source_features = collect_feature(
            self.source_loader, self.model.backbone, config.device, self.feat_dim, True
        )
        target_features = collect_feature(
            self.seq_loader, self.model.backbone, config.device, self.feat_dim, False
        )
        num_s = len(source_features)
        reduced_feats, _ = reduce_dimension(
            torch.cat((source_features, target_features), dim=0).cpu().numpy(),
            mode=config.dim_reduction,
            dim=config.reduced_dim
        )
        self.reduced_feats_s = reduced_feats[:num_s, :]
        self.reduced_feats_t = reduced_feats[num_s:, :]

    def update_pseudo_labels(self, epoch):
        config = self.config
        if epoch == self.config.consistency_start_epoch:
            # use pretrained teacher
            teacher_model = initialize_model(config, self.num_classes).to(config.device)
            load(teacher_model, config.teacher_model_path, device=config.device)
            self.logger.write("\nLoading from %s...\n" % config.teacher_model_path)
        else:
            # load current best
            featurizer, classifier = initialize_model(
                config, d_out=self.num_classes, is_featurizer=True
            )
            teacher_model = RegDNetwork(
                featurizer, classifier,
                bottleneck_dim=config.bottleneck_dim,
                bottleneck_cluster=config.bottleneck_cluster,
                use_additional_u_head=(config.w_compat > 0.0),
                dropout=0.0, num_classes=self.num_classes
            ).to(config.device)
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
        device = self.device
        config = self.config
        if self.is_training:
            (x_s, x_s_u), labels_s, metadata = batch
            x_s_u = x_s_u.to(device)
        else:
            x_s, labels_s, metadata = batch
        ns = x_s.shape[0]
        x_s = x_s.to(device)
        labels_s = labels_s.to(device)
        g = move_to(self.grouper.metadata_to_group(metadata), self.device)

        if unlabeled_batch is None:
            # Validataion
            results = {}
            results["ns"] = ns
            results["y_pred"] = self.model(x_s)
            results["y_true"] = labels_s
            results["g"] = g
            results["metadata"] = metadata
            results["idx_s"] = metadata[:, -1].long()
            return results
        
        # Training
        (x_t, x_t_u), pseudo_labels, metadata_t = unlabeled_batch
        x_t = x_t.to(device)
        x_t_u = x_t_u.to(device)
        pseudo_labels = pseudo_labels.to(device)
        labels_t_scrambled = torch.ones_like(pseudo_labels).to(device) + self.num_classes
        g_unlab = move_to(self.grouper.metadata_to_group(metadata_t), self.device)

        x = torch.cat((x_s, x_t), dim=0)
        x_u = torch.cat((x_s_u, x_t_u), dim=0)

        o = self.model(x, True)
        o_u = self.model(x_u, True)

        results = {
            "ns": ns,
            "labels_s": labels_s,
            "pseudo_labels": pseudo_labels,
            "labels_t_scrambled": labels_t_scrambled,
            "o": o,
            "o_u": o_u,
            "g": g,
            "unlabeled_g": g_unlab,
            "pseudolabels_kept_frac": 1.0,
            "y_pred": o["y"][:ns],
            "y_true": labels_s,
            "metadata": metadata,
            "unlabeled_metadata": metadata_t,
            "idx_s": metadata[:, -1].long(),
            "idx_t": metadata_t[:, -1].long()
        }
        if self.do_reduce_dim:
            results["reduced_feats_s"] = self.reduced_feats_s[results["idx_s"]]
            results["reduced_feats_s"] = torch.from_numpy(results["reduced_feats_s"]).to(device)
            results["reduced_feats_t"] = self.reduced_feats_t[results["idx_t"]]
            results["reduced_feats_t"] = torch.from_numpy(results["reduced_feats_t"]).to(device)
            
        # Generate online pseudo label if required
        if not config.noisy_student:
            y_t = o["y"][ns:]
            with torch.no_grad():
                _, pseudolabels, pseudolabels_kept_frac, mask = self.process_pseudolabels_function(
                    y_t,
                    self.confidence_threshold,
                )
                results["pseudo_labels"] = detach_and_clone(pseudolabels)
                results["mask"] = mask
                self.save_metric_for_logging(
                    results, "pseudolabels_kept_frac", pseudolabels_kept_frac
                )  
        return results

    def objective(self, results):
        if self.is_training:
            self.iteration += 1
        else:
            return self.loss.compute(results["y_pred"], results["y_true"], return_dict=False)
        
        if not self.is_classification:
            return self.objective_regression(results)

        args = self.config
        ns = results["ns"]

        o = results["o"]
        o_u = results["o_u"]
        y, y_alt, y_nograd, y_alt_nograd =\
            o["y"], o["y_cluster_all"], o["y_nograd"], o["y_cluster_all_nograd"]
        f, bf = o["feature"], o["bottleneck_feature"]
        y_u, y_u_alt, y_u_nograd, y_u_alt_nograd =\
            o_u["y"], o_u["y_cluster_all"], o_u["y_nograd"], o_u["y_cluster_all_nograd"]
        f_u, bf_u = o_u["feature"], o_u["bottleneck_feature"]
        pseudo_labels = results["pseudo_labels"]
        labels_s = results["labels_s"]
        labels_t_scrambled = results["labels_t_scrambled"]
        mask = results["mask"] if "mask" in results else None

        f_s, f_t = f[:ns], f[ns:]    # Weak Aug: label features, unlabel features
        f_s_u, f_t_u = f_u[:ns], f_u[ns:]  # Strong Aug: label features, unlabel features
        y_s, y_t = y[:ns], y[ns:]        # Weak Aug, source head: label preds, unlabel preds
        y_s_u, y_t_u = y_u[:ns], y_u[ns:]  # Strong Aug, source head: label preds, unlabel preds
        y_s_alt, y_t_alt = y_alt[:ns], y_alt[ns:]    # Weak Aug, target head: label preds, unlabel preds
        y_s_u_alt, y_t_u_alt = y_u_alt[:ns], y_u_alt[ns:]  # Strong Aug, target head: label preds, unlabel preds
        bf_s, bf_t = bf[:ns], bf[ns:]         # Weak Aug: label bottleneck features, unlabel bottleneck features
        bf_s_u, bf_t_u = bf_u[:ns], bf_u[ns:]   # Strong Aug: label bottleneck features, unlabel bottleneck features

        # Nograd outputs
        y_s_nograd, y_t_nograd = y_nograd[:ns], y_nograd[ns:]        # Weak Aug, source head: label preds, unlabel preds
        y_s_u_nograd, y_t_u_nograd = y_u_nograd[:ns], y_u_nograd[ns:]  # Strong Aug, source head: label preds, unlabel preds
        y_s_alt_nograd, y_t_alt_nograd = y_alt_nograd[:ns], y_alt_nograd[ns:]    # Weak Aug, target head: label preds, unlabel preds
        y_s_u_alt_nograd, y_t_u_alt_nograd = y_u_alt_nograd[:ns], y_u_alt_nograd[ns:]  # Strong Aug, target head: label preds, unlabel preds

        # U only cluster outputs
        if args.w_compat > 0.0:
            y_alt2, y_alt2_nograd = o["y_cluster_u"], o["y_cluster_u_nograd"]
            y_u_alt2, y_u_alt2_nograd = o_u["y_cluster_u"], o_u["y_cluster_u_nograd"]
            y_s_alt2, y_t_alt2 = y_alt2[:ns], y_alt2[ns:]
            y_s_u_alt2, y_t_u_alt2 = y_u_alt2[:ns], y_u_alt2[ns:]
            y_s_alt2_nograd, _ = y_alt2_nograd[:ns], y_alt2_nograd[ns:]
            y_s_u_alt2_nograd, _ = y_u_alt2_nograd[:ns], y_u_alt2_nograd[ns:]
        
        # dimension reduction
        if self.do_reduce_dim:
            f_s_cluster = results["reduced_feats_s"]
            f_t_cluster = results["reduced_feats_t"]
        else:
            f_s_cluster = f_s
            f_t_cluster = f_t

        # ERM
        cls_loss = self.loss.compute(y_s_u, labels_s, return_dict=False)

        # Consistency loss
        if self.config.w_con != 0.0:
            consistency_loss = self.unlabeled_loss.compute(
                    y_t_u if mask is None else y_t_u[mask],
                    pseudo_labels,
                    return_dict=False,
            ) * results['pseudolabels_kept_frac']
        else:
            consistency_loss = 0.0

        # Cluster loss
        preds1_u = torch.cat((y_s_alt_nograd, y_t_alt_nograd), dim=0)
        preds2_u = torch.cat((y_s_u_alt_nograd, y_t_u_alt_nograd), dim=0)

        inputs = {
            "x1": torch.cat((f_s_cluster, f_t_cluster), dim=0),
            "preds1_u": preds1_u,
            "preds2_u": preds2_u,
            "labels": torch.cat((labels_s, labels_t_scrambled), dim=0),
        }
        bce_loss, sim_matrix_all = self.cluster_loss.compute_losses(inputs, include_label=False)
        max_prob_alt, pseudo_labels_alt = torch.max(F.softmax(y_t_alt, dim=-1), dim=-1)


        # Disentangle loss
        if args.w_supcon != 0.0 or args.w_irm != 0.0:
            clusters_s_prob = F.softmax(y_s_alt, dim=-1)
            sup_con_loss, irm_loss = self.disentangle_loss(
                x_s=torch.cat((bf_s.unsqueeze(1), bf_s_u.unsqueeze(1)), dim=1),
                labels_s=labels_s,
                clusters_s=clusters_s_prob,
                x_t=torch.cat((bf_t.unsqueeze(1), bf_t_u.unsqueeze(1)), dim=1),
                update_memory=True
            )
        else:
            sup_con_loss, irm_loss = 0.0, 0.0

        # Compatibility loss
        compute_compat = args.w_compat != 0.0
        # Get unlabel sample mask and similarity matrix
        if compute_compat and (not args.cluster_all_compat):
            # clustering with unlabel data only
            inputs = {
                "x1": torch.cat((f_s_cluster, f_t_cluster), dim=0),
                "preds1_u": y_alt2 if self.back_cluster else y_alt2_nograd,
                "preds2_u": y_u_alt2 if self.back_cluster else y_u_alt2_nograd,
                "labels": torch.cat((labels_s, labels_t_scrambled), dim=0),
            }
            bce_loss_u, sim_matrix_ulb = self.cluster_loss.compute_losses(
                inputs, unlabel_only=True, matrix_only=self.is_regression
            )
            bce_loss += bce_loss_u
            # max_prob_alt2, pseudo_labels_alt2 = torch.max(F.softmax(y_t_alt2, dim=1), dim=-1)

        if compute_compat:
            if args.back_compat_l:
                p_t_nograd = F.softmax(y_t, dim=1)
                p_t_u_nograd = F.softmax(y_t_u, dim=1)
            else:
                p_t_nograd = F.softmax(y_t_nograd, dim=1)
                p_t_u_nograd = F.softmax(y_t_u_nograd, dim=1)
            if args.back_compat_u:
                p_s_alt_nograd = F.softmax(y_s_alt, dim=1)\
                    if args.cluster_all_compat else F.softmax(y_s_alt2, dim=1)
                p_s_u_alt_nograd = F.softmax(y_s_u_alt, dim=1)\
                    if args.cluster_all_compat else F.softmax(y_s_u_alt2, dim=1)
            else:
                p_s_alt_nograd = F.softmax(y_s_alt_nograd, dim=1)\
                    if args.cluster_all_compat else F.softmax(y_s_alt2_nograd, dim=1)
                p_s_u_alt_nograd = F.softmax(y_s_u_alt_nograd, dim=1)\
                    if args.cluster_all_compat else F.softmax(y_s_u_alt2_nograd, dim=1)

        low_t, high_t, low_p, high_p = 0.0, 0.0, 0.0, 0.0
        irm_low_p = 0.0
        if compute_compat:
            # refine unlabel similarity matrix
            cluster_logits = y_t_alt if args.cluster_all_compat else y_t_alt2
            sim_matrix_ulb_refined, low_t, high_t, low_p, high_p = get_ulb_sim_matrix(
                args.compat_mode, sim_matrix_ulb, labels_t_scrambled, cluster_logits, bf_t, self.device,
                sim_threshold=args.compat_sim_threshold, diff_threshold=args.compat_diff_threshold,
                sim_ratio=self.compat_sim_ratio, diff_ratio=self.compat_diff_ratio
            )
            cluster_entropy = calc_entropy(cluster_logits)
            batch_diversity, global_diversity = calc_diversity(cluster_logits, self.num_classes)

            # l head compat with u
            pairs1, pairs2_weak = PairEnum(p_t_nograd)
            _, pairs2 = PairEnum(p_t_u_nograd)
            lu_compatibility_loss = self.bce(pairs1, pairs2, sim_matrix_ulb_refined)

            # u head compat with l
            labels_s_view = labels_s.contiguous().view(-1, 1)
            sim_matrix_lb = torch.eq(labels_s_view, labels_s_view.T).float().to(self.device)
            sim_matrix_lb = (sim_matrix_lb - 0.5) * 2.0  # same label=1.0, diff label=-1.0
            pairs1, _ = PairEnum(p_s_alt_nograd)
            _, pairs2 = PairEnum(p_s_u_alt_nograd)
            ul_compatibility_loss = self.bce(pairs1, pairs2, sim_matrix_lb.flatten())
            compatibility_loss = lu_compatibility_loss + args.w_compat_u * ul_compatibility_loss

            # compat IRM loss
            if args.compat_irm_sim_mode == 'stats':
                sim_matrix_ulb_full, _, _, _, _ = get_ulb_sim_matrix(
                    'stats', sim_matrix_ulb, labels_t_scrambled, cluster_logits, bf_t, self.device,
                    sim_threshold=-2.0, diff_threshold=-2.0,
                    sim_ratio=0.06, diff_ratio=0.6, update_list=(args.compat_mode=='stats')
                )
            elif args.compat_irm_sim_mode == 'prob':
                sim_matrix_ulb_full, _, _, _, _ = get_ulb_sim_matrix(
                    'prob', sim_matrix_ulb, labels_t_scrambled, cluster_logits, bf_t, self.device,
                    sim_threshold=-2.0, diff_threshold=-2.0,
                    sim_ratio=0.7 / self.num_classes, diff_ratio=0.7*(self.num_classes-1)/self.num_classes,
                    update_list=(args.compat_mode=='stats')
                )
            else:
                assert False
            irm_low_p = float((sim_matrix_ulb_full<0.0).sum()) / sim_matrix_ulb_full.nelement()
            p_t, p_t_u = F.softmax(y_t, dim=1), F.softmax(y_t_u, dim=1)
            pairs1, _ = PairEnum(p_t)
            _, pairs2 = PairEnum(p_t_u)
            compat_irm_t = self.bce(pairs1, pairs2, sim_matrix_ulb_full)
            p_s, p_s_u = F.softmax(y_s, dim=1), F.softmax(y_s_u, dim=1)
            pairs1, _ = PairEnum(p_s)
            _, pairs2 = PairEnum(p_s_u)
            compat_irm_s = self.bce(pairs1, pairs2, sim_matrix_lb.flatten())
            compat_irm_loss = torch.var(torch.stack([compat_irm_s, compat_irm_t]))
        else:
            compatibility_loss = 0.0
            cluster_entropy = 0.0
            batch_diversity, global_diversity = 0.0, 0.0

        # Transfer loss
        if self.is_classification and self.w_transfer != 0.0:
            transfer_loss = self.ts(y_t)
        else:
            transfer_loss = 0.0

        loss = self.w_transfer * transfer_loss\
            + self.w_cluster * bce_loss\
            + self.w_sup * sup_con_loss + self.w_irm * irm_loss\
            + self.w_erm * cls_loss\
            + self.w_compat * compatibility_loss\
            + self.w_con * consistency_loss\
            + self.w_compat_irm * compat_irm_loss

        # Plot on tensorboard
        if self.is_training:
            visualize_items = {
                "Transfer loss": transfer_loss,
                "Consistency loss": consistency_loss * self.w_con,
                "Supervised contrastive loss": sup_con_loss * self.w_sup,
                "IRM loss": irm_loss * self.w_irm,
                "Compatibility loss": compatibility_loss * self.w_compat,
                "ERM loss": cls_loss * self.w_erm,
                "Cluster loss": bce_loss * self.w_cluster,
                "Compat IRM loss": compat_irm_loss,
                "Pseudo label kept frac": results["pseudolabels_kept_frac"],
                "Compat low threshold": low_t,
                "Compat high threshold": high_t,
                "Compat low percentage": low_p,
                "Compat high percentage": high_p,
                "Compat IRM low percentage": irm_low_p,
                "Compat cluster entropy": cluster_entropy,
                "Compat batch diversity": batch_diversity,
                "Compat global diversity": global_diversity,
                "Total loss": loss
            }
            self.visualizer.plot_items(self.iteration, visualize_items)

        return loss

    def objective_regression(self, results):
        args = self.config
        ns = results["ns"]
        o = results["o"]
        o_u = results["o_u"]
        y, y_alt, y_nograd, y_alt_nograd =\
            o["y"], o["y_cluster_all"], o["y_nograd"], o["y_cluster_all_nograd"]
        f, bf = o["feature"], o["bottleneck_feature"]
        y_u, y_u_alt, y_u_nograd, y_u_alt_nograd =\
            o_u["y"], o_u["y_cluster_all"], o_u["y_nograd"], o_u["y_cluster_all_nograd"]
        f_u, bf_u = o_u["feature"], o_u["bottleneck_feature"]
        pseudo_labels = results["pseudo_labels"]
        labels_s = results["labels_s"]
        labels_t_scrambled = results["labels_t_scrambled"]
        mask = results["mask"] if "mask" in results else None

        f_s, f_t = f[:ns], f[ns:]    # Weak Aug: label features, unlabel features
        f_s_u, f_t_u = f_u[:ns], f_u[ns:]  # Strong Aug: label features, unlabel features
        y_s, y_t = y[:ns], y[ns:]        # Weak Aug, source head: label preds, unlabel preds
        y_s_u, y_t_u = y_u[:ns], y_u[ns:]  # Strong Aug, source head: label preds, unlabel preds
        y_s_alt, y_t_alt = y_alt[:ns], y_alt[ns:]    # Weak Aug, target head: label preds, unlabel preds
        y_s_u_alt, y_t_u_alt = y_u_alt[:ns], y_u_alt[ns:]  # Strong Aug, target head: label preds, unlabel preds
        bf_s, bf_t = bf[:ns], bf[ns:]         # Weak Aug: label bottleneck features, unlabel bottleneck features
        bf_s_u, bf_t_u = bf_u[:ns], bf_u[ns:]   # Strong Aug: label bottleneck features, unlabel bottleneck features

        # Nograd outputs
        y_s_nograd, y_t_nograd = y_nograd[:ns], y_nograd[ns:]        # Weak Aug, source head: label preds, unlabel preds
        y_s_u_nograd, y_t_u_nograd = y_u_nograd[:ns], y_u_nograd[ns:]  # Strong Aug, source head: label preds, unlabel preds
        y_s_alt_nograd, y_t_alt_nograd = y_alt_nograd[:ns], y_alt_nograd[ns:]    # Weak Aug, target head: label preds, unlabel preds
        y_s_u_alt_nograd, y_t_u_alt_nograd = y_u_alt_nograd[:ns], y_u_alt_nograd[ns:]  # Strong Aug, target head: label preds, unlabel preds

        # U only cluster outputs
        if args.w_compat > 0.0:
            y_alt2, y_alt2_nograd = o["y_cluster_u"], o["y_cluster_u_nograd"]
            y_u_alt2, y_u_alt2_nograd = o_u["y_cluster_u"], o_u["y_cluster_u_nograd"]
            y_s_alt2, y_t_alt2 = y_alt2[:ns], y_alt2[ns:]
            y_s_u_alt2, y_t_u_alt2 = y_u_alt2[:ns], y_u_alt2[ns:]
            y_s_alt2_nograd, _ = y_alt2_nograd[:ns], y_alt2_nograd[ns:]
            y_s_u_alt2_nograd, _ = y_u_alt2_nograd[:ns], y_u_alt2_nograd[ns:]
        
        # dimension reduction
        if self.do_reduce_dim:
            f_s_cluster = results["reduced_feats_s"]
            f_t_cluster = results["reduced_feats_t"]
        else:
            f_s_cluster = f_s
            f_t_cluster = f_t

        # ERM
        cls_loss = self.loss.compute(y_s_u, labels_s, return_dict=False)

        # Consistency loss
        if self.config.w_con != 0.0:
            consistency_loss = self.unlabeled_loss.compute(
                    y_t_u if mask is None else y_t_u[mask],
                    pseudo_labels,
                    return_dict=False,
            ) * results['pseudolabels_kept_frac']
        else:
            consistency_loss = 0.0

        # Cluster loss (ulb only)
        labels_s_cluster = torch.zeros(labels_s.shape[0]).to(self.device)
        inputs = {
            "x1": torch.cat((f_s_cluster, f_t_cluster), dim=0),
            "preds1_u": y_alt2 if self.back_cluster else y_alt2_nograd,
            "preds2_u": y_u_alt2 if self.back_cluster else y_u_alt2_nograd,
            "labels": torch.cat((labels_s_cluster, labels_t_scrambled), dim=0),
        }
        bce_loss, sim_matrix_ulb = self.cluster_loss.compute_losses(inputs, unlabel_only=True)
        max_prob_alt, pseudo_labels_alt = torch.max(F.softmax(y_t_alt2, dim=-1), dim=-1)

        # Disentangle loss
        sup_con_loss, irm_loss = 0.0, 0.0

        # Compatibility loss
        compute_compat = args.w_compat != 0.0
        y_t_compat = y_t if args.back_compat_l else y_t_nograd
        y_t_u_compat = y_t_u if args.back_compat_l else y_t_u_nograd
        y_s_compat = y_s_alt2 if args.back_compat_u else y_s_alt2_nograd
        y_s_u_compat = y_s_u_alt2 if args.back_compat_u else y_s_u_alt2_nograd

        low_t, high_t, low_p, high_p = 0.0, 0.0, 0.0, 0.0
        irm_low_p = 0.0
        if compute_compat:
            # refine unlabel similarity matrix
            cluster_logits = y_t_alt2
            sim_matrix_ulb_refined, low_t, high_t, low_p, high_p = get_ulb_sim_matrix(
                args.compat_mode, sim_matrix_ulb, labels_t_scrambled, cluster_logits, bf_t, self.device,
                sim_threshold=args.compat_sim_threshold, diff_threshold=args.compat_diff_threshold,
                sim_ratio=self.compat_sim_ratio, diff_ratio=self.compat_diff_ratio
            )

            lu_compatibility_loss = self.get_regression_compat_loss(
                y_t_compat, y_t_u_compat, sim_matrix_ulb_refined,
                sim_margin=args.compat_sim_margin, diff_margin=args.compat_margin
            )

            sim_matrix_lb = self.get_lb_sim_matrix(labels_s)
            pairs1, _ = PairEnum(F.softmax(y_s_compat, dim=1))
            _, pairs2 = PairEnum(F.softmax(y_s_u_compat, dim=1))
            ul_compatibility_loss = self.bce(pairs1, pairs2, sim_matrix_lb)

            compatibility_loss = lu_compatibility_loss + args.w_compat_u * ul_compatibility_loss
            cluster_entropy = 0.0
            batch_diversity, global_diversity = 0.0, 0.0
        else:
            compatibility_loss = 0.0
            cluster_entropy = 0.0
            batch_diversity, global_diversity = 0.0, 0.0

        # compat IRM loss
        if args.compat_irm_sim_mode == 'stats':
            sim_matrix_ulb_full= sim_matrix_ulb
        elif args.compat_irm_sim_mode == 'prob':
            sim_matrix_ulb_full, _, _, _, _ = get_ulb_sim_matrix(
                'prob', sim_matrix_ulb, labels_t_scrambled, cluster_logits, bf_t, self.device,
                sim_threshold=-2.0, diff_threshold=-2.0,
                sim_ratio=0.7 / self.num_classes, diff_ratio=0.7*(self.num_classes-1)/self.num_classes,
                update_list=(args.compat_mode=='stats')
            )
        else:
            assert False
        irm_low_p = float((sim_matrix_ulb_full<0.0).sum()) / sim_matrix_ulb_full.nelement()

        compat_irm_t = self.get_regression_compat_loss(
            y_t, y_t_u, sim_matrix_ulb_full,
            sim_margin=args.compat_sim_margin, diff_margin=args.compat_margin
        )
        compat_irm_s = self.get_regression_compat_loss(
            y_s, y_s_u, sim_matrix_lb,
            sim_margin=args.compat_sim_margin, diff_margin=args.compat_margin
        )
        compat_irm_loss = torch.var(torch.stack([compat_irm_s, compat_irm_t]))
        lb_low_p = float((sim_matrix_lb<0.0).sum()) / sim_matrix_lb.nelement()

        # Transfer loss
        if self.is_classification and self.w_transfer != 0.0:
            transfer_loss = self.ts(y_t)
        else:
            transfer_loss = 0.0

        loss = self.w_transfer * transfer_loss\
            + self.w_cluster * bce_loss\
            + self.w_sup * sup_con_loss + self.w_irm * irm_loss\
            + self.w_erm * cls_loss\
            + self.w_compat * compatibility_loss\
            + self.w_con * consistency_loss\
            + self.w_compat_irm * compat_irm_loss

        # Plot on tensorboard
        if self.is_training:
            visualize_items = {
                "Transfer loss": transfer_loss,
                "Consistency loss": consistency_loss * self.w_con,
                "Supervised contrastive loss": sup_con_loss * self.w_sup,
                "IRM loss": irm_loss * self.w_irm,
                "Compatibility loss": compatibility_loss * self.w_compat,
                "ERM loss": cls_loss * self.w_erm,
                "Cluster loss": bce_loss * self.w_cluster,
                "Compat IRM loss": compat_irm_loss,
                "Pseudo label kept frac": results["pseudolabels_kept_frac"],
                "Compat low threshold": low_t,
                "Compat high threshold": high_t,
                "Compat low percentage": low_p,
                "Compat high percentage": high_p,
                "Compat IRM low percentage": irm_low_p,
                "Compat label low percentage": lb_low_p,
                "Compat cluster entropy": cluster_entropy,
                "Compat batch diversity": batch_diversity,
                "Compat global diversity": global_diversity,
                "Total loss": loss
            }
            self.visualizer.plot_items(self.iteration, visualize_items)
        return loss

    def get_lb_sim_matrix(self, labels_s):
        p1, p2 = PairEnum(labels_s)
        sim_matrix_lb = (torch.abs(p1 - p2) < self.config.compat_margin).float()
        sim_matrix_lb = (sim_matrix_lb - 0.5) * 2.0  # same label=1.0, diff label=-1.0
        return sim_matrix_lb.squeeze(1)

    def get_regression_compat_loss(self, y, y_u, sim_matrix, sim_margin, diff_margin):
        pairs1, _ = PairEnum(y)
        _, pairs2 = PairEnum(y_u)
        loss = torch.zeros_like(sim_matrix).to(self.device)
        mse = nn.MSELoss(reduction='none')
        if (sim_matrix > 0).sum() > 0:
            loss_sim = mse(pairs1[sim_matrix > 0], pairs2[sim_matrix > 0]).mean(dim=1)
            loss_sim[loss_sim < sim_margin * sim_margin] = 0
            loss[sim_matrix > 0] = loss_sim

        if (sim_matrix < 0).sum() > 0:
            loss_diff = mse(pairs1[sim_matrix < 0], pairs2[sim_matrix < 0]).mean(dim=1)
            loss_diff[loss_diff > diff_margin * diff_margin] = 0
            loss[sim_matrix < 0] = -loss_diff
        return loss.mean()
