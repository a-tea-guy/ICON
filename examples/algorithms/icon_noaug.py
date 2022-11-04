from codecs import xmlcharrefreplace_errors
import torch
import torch.nn.functional as F
from models.initializer import initialize_model
from algorithms.ERM import ERM
from algorithms.single_model_algorithm import SingleModelAlgorithm
from scheduler import LinearScheduleWithWarmupAndThreshold
from wilds.common.utils import split_into_groups, numel
from configs.supported import process_pseudolabels_functions
import copy
from utils import load, move_to, detach_and_clone, collate_list, concat_input, Visualizer, collect_feature, reduce_dimension
from examples.models.icon_network import RegDNetwork
from examples.icon_losses import ClusterLoss, DisentanglementLoss, Normalize, BCE, get_ulb_sim_matrix, PairEnum, TsallisEntropy, calc_entropy, calc_diversity, compute_ulb_sim_matrix, BCEMultiTask
from train import infer_predictions

class RegDisentNoAug(SingleModelAlgorithm):
    """
    PseudoLabel.
    This is a vanilla pseudolabeling algorithm which updates the model per batch and incorporates a confidence threshold.

    Original paper:
        @inproceedings{lee2013pseudo,
            title={Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks},
            author={Lee, Dong-Hyun and others},
            booktitle={Workshop on challenges in representation learning, ICML},
            volume={3},
            number={2},
            pages={896},
            year={2013}
            }
    """
    def __init__(self, config, d_out, grouper, loss, metric, n_train_steps):
        featurizer, classifier = initialize_model(
            config, d_out=d_out, is_featurizer=True
        )
        self.feat_dim = featurizer.d_out
        self.num_classes = config.num_pseudo_classes if (config.dataset == 'ogb-molpcba') else d_out
        self.d_out = d_out
        self.config = config
        model = RegDNetwork(
            featurizer, classifier,
            bottleneck_dim=config.bottleneck_dim,
            bottleneck_cluster=config.bottleneck_cluster,
            use_additional_u_head=(config.w_compat > 0.0),
            dropout=config.regd_dropout, num_classes=self.num_classes
        )
        # initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        # algorithm hyperparameters
        con_start_iter = float(n_train_steps) / float(config.n_epochs) * config.consistency_start_epoch
        self.lambda_scheduler = LinearScheduleWithWarmupAndThreshold(
            max_value=1.0,
            step_every_batch=True, # step per batch
            last_warmup_step=con_start_iter,
            threshold_step=con_start_iter + config.pseudolabel_T2 * n_train_steps
        )
        self.schedulers.append(self.lambda_scheduler)
        self.scheduler_metric_names.append(None)
        self.confidence_threshold = config.self_training_threshold
        if config.process_pseudolabels_function is not None:
            self.process_pseudolabels_function = process_pseudolabels_functions[config.process_pseudolabels_function]
        
        self.l2norm = Normalize(2)
        self.bce = BCE()
        self.ts = TsallisEntropy(
            temperature=config.ts_temperature, alpha=config.ts_alpha
        )
        self.back_cluster = False
        self.compat_sim_ratio = config.compat_sim_ratio
        self.compat_diff_ratio = config.compat_diff_ratio
        # initialize cluster and disentangle loss
        self.cluster_loss = ClusterLoss(
            self.device, self.num_classes, config.bce_type,
            config.cosine_threshold, config.topk
        )
        self.disentangle_loss = DisentanglementLoss(
            self.num_classes, temperature=config.supcon_temperature,
            use_target_as_positive=False,
            memory_length_per_class=500
        )

        # Additional logging
        self.visualizer = Visualizer(config.log_dir)
        self.iteration = 0
        # Task specific parameters
        self.is_regression = (config.dataset == "poverty")
        self.is_classification = not self.is_regression
        self.is_multitask = (config.dataset == 'ogb-molpcba')

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
        self.w_erm = 1.0 if epoch >= args.erm_start_epoch else 0.0
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
        source_features, source_labels = collect_feature(
            self.source_loader, self.model.backbone, config.device, self.feat_dim,
            two_views=False, has_labels=True
        )
        #torch.save(source_features, "src_mlm.obj")
        #torch.save(source_labels, "labels_mlm.obj")
        target_features = collect_feature(
            self.seq_loader, self.model.backbone, config.device, self.feat_dim,
            two_views=False, has_labels=False
        )
        #torch.save(target_features, "tgt_mlm.obj")
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
            teacher_model = initialize_model(config, self.d_out).to(config.device)
            load(teacher_model, config.teacher_model_path, device=config.device)
            self.logger.write("\nLoading from %s...\n" % config.teacher_model_path)
        else:
            # load current best
            featurizer, classifier = initialize_model(
                config, d_out=self.d_out, is_featurizer=True
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
        """
        Overrides single_model_algorithm.process_batch().
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
            - unlabeled_batch (tuple of Tensors or None): a batch of data yielded by unlabeled data loader
        Output:
            - results (dictionary): information about the batch
                - y_true (Tensor): ground truth labels for batch
                - g (Tensor): groups for batch
                - metadata (Tensor): metadata for batch
                - y_pred (Tensor): model output for batch
                - unlabeled_g (Tensor): groups for unlabeled batch
                - unlabeled_metadata (Tensor): metadata for unlabeled batch
                - unlabeled_y_pseudo (Tensor): pseudolabels on the unlabeled batch, already thresholded 
                - unlabeled_y_pred (Tensor): model output on the unlabeled batch, already thresholded 
        """
        # Labeled examples
        x, y_true, metadata = batch
        n_lab = len(metadata)
        x = move_to(x, self.device)
        y_true = move_to(y_true, self.device)
        g = move_to(self.grouper.metadata_to_group(metadata), self.device)
        device = self.device

        # package the results
        results = {
            'g': g,
            'y_true': y_true,
            'metadata': metadata
        }

        if unlabeled_batch is not None:
            if self.config.noisy_student:
                x_unlab, pseudolabels, metadata_unlab = unlabeled_batch
                pseudolabels = pseudolabels.to(device)
                pseudolabels = detach_and_clone(pseudolabels)
                pseudolabels_kept_frac = 1.0
                mask = None
            else:
                x_unlab, metadata_unlab = unlabeled_batch
            x_unlab = move_to(x_unlab, self.device)
            g_unlab = move_to(self.grouper.metadata_to_group(metadata_unlab), self.device)
            labels_t_scrambled = torch.ones(len(metadata_unlab)).to(self.device) + self.num_classes
            # results['unlabeled_metadata'] = metadata_unlab
            # results['unlabeled_g'] = g_unlab
            x_cat = concat_input(x, x_unlab)
            outputs = self.model(x_cat, True)
            y_t = outputs["y"][n_lab:]
            
            if not self.config.noisy_student:
                unlabeled_y_pred, pseudolabels, pseudolabels_kept_frac, mask = self.process_pseudolabels_function(
                    y_t,
                    self.confidence_threshold
                )
                pseudolabels = detach_and_clone(pseudolabels)
            
            results = {
                "ns": n_lab,
                "labels_s": y_true,
                "pseudo_labels": pseudolabels,
                "labels_t_scrambled": labels_t_scrambled,
                "o": outputs,
                "g": g,
                "unlabeled_g": g_unlab,
                "y_pred": outputs["y"][:n_lab],
                "y_true": y_true,
                "metadata": metadata,
                "unlabeled_metadata": metadata_unlab,
                "idx_s": metadata[:, -1].long(),
                "idx_t": metadata_unlab[:, -1].long(),
            }
            if mask is not None:
                results["mask"] = mask

            if self.do_reduce_dim:
                results["reduced_feats_s"] = self.reduced_feats_s[results["idx_s"]]
                results["reduced_feats_s"] = torch.from_numpy(results["reduced_feats_s"]).to(device)
                results["reduced_feats_t"] = self.reduced_feats_t[results["idx_t"]]
                results["reduced_feats_t"] = torch.from_numpy(results["reduced_feats_t"]).to(device)
        else:
            results['y_pred'] = self.model(x)
            results["ns"] = n_lab
            results["idx_s"] = metadata[:, -1].long()
            pseudolabels_kept_frac = 0

        self.save_metric_for_logging(
            results, "pseudolabels_kept_frac", pseudolabels_kept_frac
        )
        return results

    def objective(self, results):
        if self.is_training:
            self.iteration += 1
        else:
            return self.loss.compute(results["y_pred"], results["y_true"], return_dict=False)

        args = self.config
        ns = results["ns"]
        is_multitask = (args.dataset == 'ogb-molpcba')

        o = results["o"]
        y, y_alt, y_nograd, y_alt_nograd =\
            o["y"], o["y_cluster_all"], o["y_nograd"], o["y_cluster_all_nograd"]
        f, bf = o["feature"], o["bottleneck_feature"]
        pseudo_labels = results["pseudo_labels"]
        labels_s = results["labels_s"]
        labels_t_scrambled = results["labels_t_scrambled"]
        mask = results["mask"] if "mask" in results else None

        f_s, f_t = f[:ns], f[ns:]    # Weak Aug: label features, unlabel features
        y_s, y_t = y[:ns], y[ns:]        # Weak Aug, source head: label preds, unlabel preds
        y_s_alt, y_t_alt = y_alt[:ns], y_alt[ns:]    # Weak Aug, target head: label preds, unlabel preds
        bf_s, bf_t = bf[:ns], bf[ns:]         # Weak Aug: label bottleneck features, unlabel bottleneck features
        y_s_nograd, y_t_nograd = y_nograd[:ns], y_nograd[ns:]        # Weak Aug, source head: label preds, unlabel preds
        y_s_alt_nograd, y_t_alt_nograd = y_alt_nograd[:ns], y_alt_nograd[ns:]    # Weak Aug, target head: label preds, unlabel preds
        
        # U only cluster outputs
        if args.w_compat > 0.0:
            y_alt2, y_alt2_nograd = o["y_cluster_u"], o["y_cluster_u_nograd"]
            y_s_alt2, y_t_alt2 = y_alt2[:ns], y_alt2[ns:]
            y_s_alt2_nograd, _ = y_alt2_nograd[:ns], y_alt2_nograd[ns:]

        # dimension reduction
        if self.do_reduce_dim:
            f_s_cluster = results["reduced_feats_s"]
            f_t_cluster = results["reduced_feats_t"]
        else:
            f_s_cluster = f_s
            f_t_cluster = f_t

        # Labeled loss
        cls_loss = self.loss.compute(y_s, labels_s, return_dict=False)

        # Pseudolabeled loss
        if self.config.w_con != 0.0:
            consistency_loss = self.loss.compute(
                y_t if mask is None else y_t[mask],
                pseudo_labels,
                return_dict=False,
            ) * results['pseudolabels_kept_frac']
            if args.dataset != 'ogb-molpcba':
                consistency_loss *= self.lambda_scheduler.value
        else:
            consistency_loss = 0

        # Cluster loss
        labels_s_cluster = torch.zeros(labels_s.shape[0]).to(self.device) if is_multitask else labels_s
        if self.is_classification and (args.w_supcon != 0.0 or args.w_irm != 0.0):
            preds1_u = torch.cat((y_s_alt_nograd, y_t_alt_nograd), dim=0)
            preds2_u = preds1_u
            inputs = {
                "x1": torch.cat((f_s_cluster, f_t_cluster), dim=0),
                "preds1_u": preds1_u,
                "preds2_u": preds2_u,
                "labels": torch.cat((labels_s_cluster, labels_t_scrambled), dim=0),
            }
            bce_loss, sim_matrix_all = self.cluster_loss.compute_losses(inputs, include_label=False)
            max_prob_alt, pseudo_labels_alt = torch.max(F.softmax(y_t_alt, dim=-1), dim=-1)
        else:
            bce_loss = 0.0

        # Disentangle loss
        if args.w_supcon != 0.0 or args.w_irm != 0.0:
            if self.is_classification:
                clusters_s_prob = F.softmax(y_s_alt, dim=-1)
                sup_con_loss, irm_loss = self.disentangle_loss(
                    x_s=torch.cat((bf_s.unsqueeze(1), bf_s.unsqueeze(1)), dim=1),
                    labels_s=labels_s,
                    clusters_s=clusters_s_prob,
                    x_t=torch.cat((bf_t.unsqueeze(1), bf_t.unsqueeze(1)), dim=1),
                    update_memory=True
                )
            else:
                sup_con_loss, irm_loss = self.disentangle_loss.get_regression_loss(
                    x_s=torch.cat((bf_s.unsqueeze(1), bf_s.unsqueeze(1)), dim=1),
                    labels_s=labels_s,
                    x_t=torch.cat((bf_t.unsqueeze(1), bf_t.unsqueeze(1)), dim=1),
                    topk=args.topk,
                    margin=args.supcon_margin
                )
        else:
            sup_con_loss, irm_loss = 0.0, 0.0
        
        # Compatibility loss
        compute_compat = args.w_compat != 0.0
        low_t, high_t, low_p, high_p = 0.0, 0.0, 0.0, 0.0
        irm_low_p = 0.0
        # Get unlabel sample mask and similarity matrix
        if compute_compat:
            if self.is_classification:
                # clustering with unlabel data only
                inputs = {
                    "x1": torch.cat((f_s_cluster, f_t_cluster), dim=0),
                    "preds1_u": y_alt2 if self.back_cluster else y_alt2_nograd,
                    "preds2_u": y_alt2 if self.back_cluster else y_alt2_nograd,
                    "labels": torch.cat((labels_s_cluster, labels_t_scrambled), dim=0),
                }
                bce_loss_u, sim_matrix_ulb = self.cluster_loss.compute_losses(
                    inputs, unlabel_only=True, matrix_only=self.is_regression
                )
                bce_loss += bce_loss_u

                if is_multitask:
                    normalize = torch.nn.Sigmoid() if args.compat_multitask_normalize else torch.nn.Identity()
                else:
                    normalize = torch.nn.Softmax(dim=1)
                if args.back_compat_l:
                    p_t_nograd = normalize(y_t)
                    p_t_u_nograd = normalize(y_t)
                else:
                    p_t_nograd = normalize(y_t_nograd)
                    p_t_u_nograd = normalize(y_t_nograd)
                if args.back_compat_u:
                    p_s_alt_nograd = normalize(y_s_alt2)
                    p_s_u_alt_nograd = normalize(y_s_alt2)
                else:
                    p_s_alt_nograd = normalize(y_s_alt2_nograd)
                    p_s_u_alt_nograd = normalize(y_s_alt2_nograd)

                # refine unlabel similarity matrix
                cluster_logits = y_t_alt if args.cluster_all_compat else y_t_alt2
                sim_matrix_ulb_refined, low_t, high_t, low_p, high_p = get_ulb_sim_matrix(
                    args.compat_mode, sim_matrix_ulb, labels_t_scrambled, cluster_logits, bf_t, self.device,
                    sim_threshold=args.compat_sim_threshold, diff_threshold=args.compat_diff_threshold,
                    sim_ratio=self.compat_sim_ratio, diff_ratio=self.compat_diff_ratio
                )
                with torch.no_grad():
                    cluster_entropy = calc_entropy(cluster_logits)
                    batch_diversity, global_diversity = calc_diversity(cluster_logits, self.num_classes)

                # l head compat with u
                pairs1, pairs2_weak = PairEnum(p_t_nograd)
                _, pairs2 = PairEnum(p_t_u_nograd)
                lu_compatibility_loss = self.get_bce_loss(pairs1, pairs2, sim_matrix_ulb_refined)

                # u head compat with l
                if args.w_compat_u != 0.0:
                    sim_matrix_lb = self.get_lb_sim_matrix(labels_s)
                    pairs1, _ = PairEnum(p_s_alt_nograd)
                    _, pairs2 = PairEnum(p_s_u_alt_nograd)
                    ul_compatibility_loss = self.get_bce_loss(pairs1, pairs2, sim_matrix_lb)
                else:
                    ul_compatibility_loss = 0.0
                compatibility_loss = lu_compatibility_loss + args.w_compat_u * ul_compatibility_loss

                # Compat IRM loss
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
                p_t, p_t_u = normalize(y_t), normalize(y_t)
                pairs1, _ = PairEnum(p_t)
                _, pairs2 = PairEnum(p_t_u)
                compat_irm_t = self.get_bce_loss(pairs1, pairs2, sim_matrix_ulb_full)
                p_s, p_s_u = normalize(y_s), normalize(y_s)
                pairs1, _ = PairEnum(p_s)
                _, pairs2 = PairEnum(p_s_u)
                sim_matrix_lb = self.get_lb_sim_matrix(labels_s)
                compat_irm_s = self.get_bce_loss(pairs1, pairs2, sim_matrix_lb)
                compat_irm_loss = torch.var(torch.stack([compat_irm_s, compat_irm_t]))
            elif self.is_regression:
                assert False
        else:
            compatibility_loss = 0.0
            cluster_entropy = 0.0
            batch_diversity, global_diversity = 0.0, 0.0
            compat_irm_loss = 0.0

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
                "Total loss": loss,
                "Lambda scheduler": self.lambda_scheduler.value,
            }
            self.visualizer.plot_items(self.iteration, visualize_items)

        return loss

    def get_bce_loss(self, pairs1, pairs2, sim_matrix):
        args = self.config
        if not self.is_multitask:
            bce_loss = self.bce(pairs1, pairs2, sim_matrix)
        else:
            if args.compat_multitask_normalize:
                bce_loss = BCEMultiTask()(pairs1, pairs2, sim_matrix)
            else:
                compat_loss_fn = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=args.compat_multitask_margin)
                pairs1 = pairs1[sim_matrix != 0.0]
                pairs2 = pairs2[sim_matrix != 0.0]
                bce_loss = compat_loss_fn(pairs1, pairs2, sim_matrix[sim_matrix!=0.0])
        return bce_loss

    def get_lb_sim_matrix(self, labels_s):
        if self.is_multitask:
            p1, p2 = PairEnum(labels_s)
            sim_matrix_lb = ((p1+p2==1.0).sum(dim=1)==0.0).float()
            sim_matrix_lb = (sim_matrix_lb - 0.5) * 2.0  # same label=1.0, diff label=-1.0
            return sim_matrix_lb
        else:
            labels_s_view = labels_s.contiguous().view(-1, 1)
            sim_matrix_lb = torch.eq(labels_s_view, labels_s_view.T).float().to(self.device)
            sim_matrix_lb = (sim_matrix_lb - 0.5) * 2.0  # same label=1.0, diff label=-1.0
            return sim_matrix_lb.flatten()