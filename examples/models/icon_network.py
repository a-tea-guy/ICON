from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm


class RegDNetwork(nn.Module):
    def __init__(self, backbone, classifier, bottleneck_dim, bottleneck_cluster=False,
                 use_additional_u_head=False, dropout=0.5, num_classes=None):
        super(RegDNetwork, self).__init__()
        self.backbone = backbone
        self.head = classifier
        self.num_classes = classifier.weight.size(0) if num_classes is None else num_classes
        if bottleneck_dim > 0:
            self.bottleneck = nn.Sequential(
                nn.Linear(backbone.d_out, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim),
                nn.ReLU()
            )
            bottleneck_out_dim = bottleneck_dim
            self.head = nn.Linear(bottleneck_out_dim, self.num_classes)
        else:
            self.bottleneck = nn.Identity()
            bottleneck_out_dim = backbone.d_out

        self.bottleneck_cluster = bottleneck_cluster
        self.use_additional_u_head = use_additional_u_head
        if bottleneck_cluster:
            self.u_head = nn.Linear(bottleneck_out_dim, self.num_classes)
        else:
            self.u_head = nn.Linear(backbone.d_out, self.num_classes)
        if use_additional_u_head:
            if bottleneck_cluster:
                self.u_only_head = nn.Linear(bottleneck_out_dim, self.num_classes)
            else:
                self.u_only_head = nn.Linear(backbone.d_out, self.num_classes)
        else:
            self.u_only_head = None
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def forward(self, x, rich_outputs=False):
        f = self.backbone(x)
        f1 = self.bottleneck(f)

        # if self.use_mask:
        #     f_clf = self.activation_map.apply(self.mask_layer) * f1
        #     f_clf_nograd = self.activation_map.apply(self.mask_layer) * f1.detach()
        # else:
        #     f_clf = f1
        #     f_clf_nograd = f1.detach()

        if self.dropout is not None:
            f1_drop = self.dropout(f1)
        else:
            f1_drop = f1
        # f1_nograd = self.bottleneck(f.detach())
        predictions = self.head(f1_drop)

        if not rich_outputs:
            return predictions
        
        preds_nograd = self.head(f1_drop.detach())
        if self.bottleneck_cluster:
            u_preds = self.u_head(f1)
            u_preds_nograd = self.u_head(f1.detach())
        else:
            u_preds = self.u_head(f)
            u_preds_nograd = self.u_head(f.detach())
        outputs = {
            "y": predictions,
            "y_cluster_all": u_preds,
            "feature": f,
            "bottleneck_feature": f1,
            "y_nograd": preds_nograd,
            "y_cluster_all_nograd": u_preds_nograd
        }
        if self.u_only_head is not None:
            if self.bottleneck_cluster:
                outputs["y_cluster_u"] = self.u_only_head(f1)
                outputs["y_cluster_u_nograd"] = self.u_only_head(f1.detach())
            else:
                outputs["y_cluster_u"] = self.u_only_head(f)
                outputs["y_cluster_u_nograd"] = self.u_only_head(f.detach())
        return outputs
    '''
    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head.parameters(), "lr": 1.0 * base_lr},
        ]
        if self.u_head is not None:
            params.append({"params": self.u_head.parameters(), "lr": 1.0 * base_lr})
        if self.u_only_head is not None:
            params.append({"params": self.u_only_head.parameters(), "lr": 1.0 * base_lr})
        return params
    '''
    