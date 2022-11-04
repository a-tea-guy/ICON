from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm


class activation_map():
    def __init__(self, activation_type):
        self.activation_type = activation_type

    def apply(self, x, soft=False):
        if self.activation_type == 'sigmoid':
            return torch.sigmoid(x)
        elif self.activation_type == 'ident':
            return x
        elif self.activation_type == 'gumbel':
            if soft:
                x_hard = F.gumbel_softmax(x, tau=1, hard=False)
            else:
                x_hard = F.gumbel_softmax(x, tau=1, hard=True)
            return x_hard[:,1].squeeze().unsqueeze(0)
        

class PCGNetwork(nn.Module):
    def __init__(self, featurizer, l_head, u_head, dropout, sup_contra, use_mlp_head=False, use_mask=False):
        super().__init__()
        self.featurizer = featurizer
        self.sup_contra = sup_contra
        self.use_mlp_head = use_mlp_head
        self.use_mask = use_mask
        self.classifier = l_head
        feat_dim = self.classifier.weight.size(1)
        num_classes = self.classifier.weight.size(0)
        if u_head is not None:
            self.u_head = u_head
        else:
            self.u_head = nn.Linear(feat_dim, num_classes)

        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
            self.use_dropout = True
        else:
            self.use_dropout = False

        if use_mask:
            mask_layer = torch.rand(feat_dim)
            self.mask_layer = torch.nn.Parameter(mask_layer)
            self.activation_map = activation_map('sigmoid')

        if sup_contra and use_mlp_head:
            self.mlp = nn.Sequential(nn.Linear(feat_dim, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, 128, bias=True))
            # self.u_mlp = nn.Sequential(nn.Linear(feat_dim, 512, bias=False), nn.BatchNorm1d(512),
            #                    nn.ReLU(inplace=True), nn.Linear(512, 128, bias=True))

    def forward(self, x):
        feature = self.featurizer(x)
        if self.use_dropout and self.training:
            feature = self.dropout(feature)
        if self.use_mask:
            feature = self.activation_map.apply(self.mask_layer) * feature
        l_out = self.classifier(feature)
        if self.u_head is not None:
            u_out = self.u_head(feature)
        else:
            u_out = l_out
        if self.sup_contra and self.use_mlp_head:
            l_proj = self.mlp(feature)
            u_proj = l_proj
        else:
            l_proj = l_out
            u_proj = u_out
        return feature, l_out, u_out, l_proj, u_proj