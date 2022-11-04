"""
This module adapts Faster-RCNN from the torchvision library to compute per-image losses,
instead of the default per-batch losses.
It is based on the version from torchvision==0.8.2,
and has not been tested on other versions.

The torchvision library is distributed under the BSD 3-Clause License:
https://github.com/pytorch/vision/blob/master/LICENSE
https://github.com/pytorch/vision/tree/master/torchvision/models/detection
"""

import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict
import torch
from torch import nn, Tensor
import warnings
from typing import Tuple, List, Dict, Optional, Union

from torch import nn
from torch.nn import functional as F
import numpy as np
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
# from torchvision.models.utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.faster_rcnn import TwoMLPHead
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork, concat_box_prediction_layers,permute_and_flatten
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from examples.icon_losses import ClusterLoss, BCE, get_ulb_sim_matrix, PairEnum, calc_entropy, calc_diversity

model_urls = {
    'fasterrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
    'fasterrcnn_mobilenet_v3_large_320_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth',
    'fasterrcnn_mobilenet_v3_large_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth'
}

def batch_concat_box_prediction_layers(box_cls, box_regression):
    # type: (List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    box_cls_flattened = []
    box_regression_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level in zip(
        box_cls, box_regression
    ):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(
            box_cls_per_level, N, A, C, H, W
        )
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(
            box_regression_per_level, N, A, 4, H, W
        )
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)

    batch_size = box_regression_flattened[0].shape[0]

    new_box_cls = []
    new_box_regression = []
    for batch_idx in range(batch_size):
        element_box_cls = [torch.unsqueeze(item[batch_idx],dim=0) for item in box_cls_flattened]
        element_box_regression = [torch.unsqueeze(item[batch_idx],dim=0) for item in box_regression_flattened]

        element_box_cls = torch.cat(element_box_cls, dim=1).flatten(0, -2)
        element_box_regression = torch.cat(element_box_regression, dim=1).reshape(-1, 4)
        new_box_cls.append(element_box_cls)
        new_box_regression.append(element_box_regression)


    return new_box_cls, new_box_regression

class RegionProposalNetworkWILDS(RegionProposalNetwork):
    def __init__(self,
                 anchor_generator,
                 head,
                 #
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 #
                 pre_nms_top_n, post_nms_top_n, nms_thresh):
        super().__init__(anchor_generator,
                            head,
                            fg_iou_thresh, bg_iou_thresh,
                            batch_size_per_image, positive_fraction,
                            pre_nms_top_n, post_nms_top_n, nms_thresh)

    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
        """
        Arguments:
            objectness (Tensor)
            pred_bbox_deltas (Tensor)
            labels (List[Tensor])
            regression_targets (List[Tensor])
        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor)
        """
        objectness, pred_bbox_deltas = batch_concat_box_prediction_layers(objectness, pred_bbox_deltas)

        objectness_loss = []
        box_loss = []

        for objectness_, regression_targets_,labels_,objectness_,pred_bbox_deltas_ in zip(objectness,regression_targets,labels,objectness,pred_bbox_deltas):

            sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(torch.unsqueeze(labels_,dim=0))
            sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
            sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]
            sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

            box_loss.append(F.smooth_l1_loss(
                pred_bbox_deltas_[sampled_pos_inds],
                regression_targets_[sampled_pos_inds],
                beta=1 / 9,
                reduction='sum',
            ) / (sampled_inds.numel()))

            objectness_loss.append(F.binary_cross_entropy_with_logits(
                objectness_[sampled_inds].flatten(), labels_[sampled_inds]
            ))

        return torch.stack(objectness_loss), torch.stack(box_loss)

    def forward(self,
                images,       # type: ImageList
                features,     # type: Dict[str, Tensor]
                targets=None  # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Tensor], Dict[str, Tensor]]
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (OrderedDict[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[Tensor]]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.
        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
            losses (Dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # RPN uses all feature maps that are available
        features = list(features.values())
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]

        raw_objectness = objectness
        raw_pred_bbox_deltas = pred_bbox_deltas
        objectness, pred_bbox_deltas = \
            concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)

        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)
        losses = {}

        if self.training:
            assert targets is not None
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                raw_objectness, raw_pred_bbox_deltas, labels, regression_targets)

            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
        return boxes, losses

def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.
    Arguments:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)
    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """
    class_logits = torch.split(class_logits, 512,dim=0)
    box_regression = torch.split(box_regression, 512,dim=0)
    classification_loss = []
    box_loss = []

    for class_logits_, box_regression_, labels_, regression_targets_ in zip(class_logits, box_regression, labels, regression_targets):
        classification_loss.append(F.cross_entropy(class_logits_, labels_))
        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.where(labels_ > 0)[0]

        labels_pos = labels_[sampled_pos_inds_subset]
        N, num_classes = class_logits_.shape

        box_regression_ = box_regression_.reshape(N, -1, 4)

        box_loss_ = F.smooth_l1_loss(
            box_regression_[sampled_pos_inds_subset, labels_pos],
            regression_targets_[sampled_pos_inds_subset],
            beta=1 / 9,
            reduction='sum',
        )
        box_loss.append(box_loss_ / labels_.numel())

    return torch.stack(classification_loss), torch.stack(box_loss)


# Compute pairwise consistency loss and IRM loss
class ICONHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.device
        self.num_classes = 2
        self.cluster_head = nn.Linear(1024, 2)
        self.cluster_loss = ClusterLoss(
            self.device, 2, config.bce_type,
            config.cosine_threshold, config.topk
        )
        self.bce = BCE()
    
    def select_samples(self, features, targets, preds1, preds2, select=512):
        # select a subset with ~same 1 and 0
        # from targets=1, select 256 samples (as much as possible)
        indices_of_1 = torch.where(targets == 1)[0]
        indices_of_0 = torch.where(targets == 0)[0]
        if (targets == 1).sum() <= select // 2:
            # select all
            indices_sel1 = indices_of_1
        else:
            indices_sel1 = np.random.choice(indices_of_1.cpu().numpy(), select // 2, replace=False)
            indices_sel1 = torch.from_numpy(indices_sel1).to(self.device)
        # from targets=0, select the rest
        indices_sel0 = np.random.choice(
            indices_of_0.cpu().numpy(), select - len(indices_sel1), replace=False
        )
        indices_sel0 = torch.from_numpy(indices_sel0).to(self.device)
        indices_sel = torch.cat((indices_sel0, indices_sel1), dim=0)
        return features[indices_sel], targets[indices_sel], preds1[indices_sel], preds2[indices_sel]

    def forward(self, features, targets, preds, preds_nograd, epoch=0):
        args = self.config

        # setup flags
        back_cluster = args.back_cluster if epoch >= args.back_cluster_start_epoch else False

        n_s = args.batch_size * 512
        f_s, f_t = features[:n_s], features[n_s:]
        y_s, y_pseudo_t = targets[:n_s], targets[n_s:]
        preds_s, preds_t = preds[:n_s], preds[n_s:]
        preds_s_nograd, preds_t_nograd = preds_nograd[:n_s], preds_nograd[n_s:]
        f_t, y_pseudo_t, preds_t, preds_t_nograd = self.select_samples(
            f_t, y_pseudo_t, preds_t, preds_t_nograd, select=512
        )
        y_t_scrambled = torch.ones(len(y_pseudo_t)).to(self.device) + 2

        # cluster loss
        preds_c = self.cluster_head(torch.cat((f_s, f_t), dim=0))
        preds_c_nograd = self.cluster_head(torch.cat((f_s, f_t), dim=0).detach())
        inputs = {
            "x1": torch.cat((f_s, f_t), dim=0),
            "preds1_u": preds_c if back_cluster else preds_c_nograd,
            "preds2_u": preds_c if back_cluster else preds_c_nograd,
            "labels": torch.cat((y_s, y_t_scrambled), dim=0),
        }
        bce_loss, sim_matrix_ulb = self.cluster_loss.compute_losses(
            inputs, unlabel_only=True, matrix_only=False
        )

        # compat loss
        # refine unlabel similarity matrix
        cluster_logits = preds_c_nograd[n_s:]
        sim_matrix_ulb_refined, low_t, high_t, low_p, high_p = get_ulb_sim_matrix(
            args.compat_mode, sim_matrix_ulb, y_t_scrambled, cluster_logits, f_t, self.device,
            sim_threshold=args.compat_sim_threshold, diff_threshold=args.compat_diff_threshold,
            sim_ratio=args.compat_sim_ratio, diff_ratio=args.compat_diff_ratio
        )
        with torch.no_grad():
            cluster_entropy = calc_entropy(cluster_logits)
            batch_diversity, global_diversity = calc_diversity(cluster_logits, self.num_classes)

        # l head compat with u
        p_t = F.softmax(preds_t, dim=-1)
        p_t_nograd = F.softmax(preds_t_nograd, dim=-1)
        p_s = F.softmax(preds_s, dim=-1)
        p_s_nograd = F.softmax(preds_s_nograd, dim=-1)
        
        p_t_compat = p_t if args.back_compat_l else p_t_nograd
        pairs1, pairs2_weak = PairEnum(p_t_compat)
        _, pairs2 = PairEnum(p_t_compat)
        lu_compatibility_loss = self.bce(pairs1, pairs2, sim_matrix_ulb_refined)
        compatibility_loss = lu_compatibility_loss

        # IRM loss
        labels_s_view = y_s.contiguous().view(-1, 1)
        sim_matrix_lb = torch.eq(labels_s_view, labels_s_view.T).float().to(self.device)
        sim_matrix_lb = (sim_matrix_lb - 0.5) * 2.0  # same label=1.0, diff label=-1.0 
        sim_matrix_lb = sim_matrix_lb.flatten()
        if args.compat_irm_sim_mode == 'stats':
            sim_matrix_ulb_full, _, _, _, _ = get_ulb_sim_matrix(
                'stats', sim_matrix_ulb, y_t_scrambled, cluster_logits, f_t, self.device,
                sim_threshold=-2.0, diff_threshold=-2.0,
                sim_ratio=0.06, diff_ratio=0.6, update_list=(args.compat_mode=='stats')
            )
        elif args.compat_irm_sim_mode == 'prob':
            sim_matrix_ulb_full, _, _, _, _ = get_ulb_sim_matrix(
                'prob', sim_matrix_ulb, y_t_scrambled, cluster_logits, f_t, self.device,
                sim_threshold=-2.0, diff_threshold=-2.0,
                sim_ratio=0.7 / self.num_classes, diff_ratio=0.7*(self.num_classes-1)/self.num_classes,
                update_list=(args.compat_mode=='stats')
            )
        elif args.compat_irm_sim_mode == 'argmax':
            y_c_t = preds_c[n_s:].argmax(dim=1).contiguous().view(-1, 1)
            sim_matrix_ulb_full = torch.eq(y_c_t, y_c_t.T).float().to(self.device)
            sim_matrix_ulb_full = (sim_matrix_ulb_full - 0.5) * 2
            sim_matrix_ulb_full = sim_matrix_ulb_full.flatten()
        irm_low_p_t = float((sim_matrix_ulb_full<0.0).sum()) / sim_matrix_ulb_full.nelement()
        irm_low_p_s = float((sim_matrix_lb<0.0).sum()) / sim_matrix_lb.nelement()
        p_t_irm = p_t if args.back_compat_irm else p_t_nograd
        p_s_irm = p_s if args.back_compat_irm else p_s_nograd
        pairs1, _ = PairEnum(p_t_irm)
        _, pairs2 = PairEnum(p_t_irm)
        compat_irm_t = self.bce(pairs1, pairs2, sim_matrix_ulb_full)
        pairs1, _ = PairEnum(p_s_irm)
        _, pairs2 = PairEnum(p_s_irm)
        compat_irm_s = self.bce(pairs1, pairs2, sim_matrix_lb)
        compat_irm_loss = torch.var(torch.stack([compat_irm_s, compat_irm_t]))

        results_dict = {
            "compat_irm_loss": compat_irm_loss,
            "compat_loss": compatibility_loss,
            "bce_loss": bce_loss,
            "cluster_entropy": cluster_entropy,
            "batch_diversity": batch_diversity,
            "irm_low_p_t": irm_low_p_t,
            "irm_low_p_s": irm_low_p_s,
            "back_cluster": back_cluster,
            "low_t": low_t,
            "low_p": low_p,
            "high_t": high_t,
            "high_p": high_p
        }
        return results_dict


class RoIHeadsWILDS(RoIHeads):
    def __init__(self, box_roi_pool, box_head, box_predictor, box_fg_iou_thresh, box_bg_iou_thresh,box_batch_size_per_image,box_positive_fraction,bbox_reg_weights,box_score_thresh,box_nms_thresh,box_detections_per_img):

        super().__init__(box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)
        self.icon_head = None
        self.epoch = 0

    def forward(self,
                features,      # type: Dict[str, Tensor]
                proposals,     # type: List[Tensor]
                image_shapes,  # type: List[Tuple[int, int]]
                targets=None   # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
                if self.has_keypoint():
                    assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'

        # here batch is maintained
        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)

        class_logits, box_regression = self.box_predictor(box_features)
        class_logits_nograd, _ = self.box_predictor(box_features.detach())
        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        additionals = None
        if self.training:
            assert labels is not None and regression_targets is not None

            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
            if self.icon_head is not None:
                icon_results = self.icon_head(
                    box_features, torch.cat(labels,dim=0),
                    class_logits, class_logits_nograd, epoch=self.epoch
                )
                additionals = icon_results

        boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                }
            )
        return result, losses, additionals

def fasterrcnn_resnet50_fpn(pretrained=False, progress=True,
                            num_classes=91, pretrained_backbone=True,
                            trainable_backbone_layers=3, config=None, **kwargs):

    assert trainable_backbone_layers <= 5 and trainable_backbone_layers >= 0
    # dont freeze any layers if pretrained model or backbone is not used
    if not (pretrained or pretrained_backbone):
        trainable_backbone_layers = 5
    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone, trainable_layers=trainable_backbone_layers)
    model = FastWILDS(backbone, 91, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['fasterrcnn_resnet50_fpn_coco'],
                                              progress=progress)
        model.load_state_dict(state_dict)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes+1)

    # create icon head
    if config is not None and config.w_compat != 0.0:
        model.roi_heads.icon_head = ICONHeads(config)
    return model

class FastWILDS(GeneralizedRCNN):
    def __init__(self, backbone, num_classes=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetworkWILDS(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)

        roi_heads = RoIHeadsWILDS(
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)


        image_mean = [0., 0., 0.] # small trick because images are already normalized
        image_std = [1., 1., 1.]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super(FastWILDS, self).__init__(backbone, rpn, roi_heads, transform)

    # Set your own forward pass
    def forward(self, images, targets=None):        
        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be passed")
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                            "of shape [N, 4], got {:}.".format(
                                                boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                        "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                        " Found invalid box {} for target at index {}."
                                        .format(degen_bb, target_idx))

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses, additionals = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        for idx, det in enumerate(detections):
            det["losses"] = {}
            for k,v in proposal_losses.items():
                det["losses"][k] = v[idx]
            for k,v in detector_losses.items():
                det["losses"][k] = v[idx]
        if additionals is None:
            return detections
        else:
            return detections, additionals

class FasterRCNNLoss(nn.Module):
    def __init__(self,device):
        self.device = device
        super().__init__()

    def forward(self, outputs, targets):
        # loss values are  loss_classifier loss_box_reg loss_objectness": loss_objectness, loss_rpn_box_reg
        try:
            elementwise_loss = torch.stack([sum(v for v in item["losses"].values()) for item in outputs])
        except:
            elementwise_loss = torch.ones(len(outputs)).to(self.device)

        return elementwise_loss
