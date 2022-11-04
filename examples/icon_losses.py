import torch
from torch import nn
import math
import torch.nn.functional as F
import numpy as np


def entropy(predictions, reduction='none'):
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == 'mean':
        return H.mean()
    else:
        return H


class TsallisEntropy(nn.Module):
    def __init__(self, temperature, alpha):
        super(TsallisEntropy, self).__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, logits):
        N, C = logits.shape
        pred = F.softmax(logits / self.temperature, dim=1) 
        entropy_weight = entropy(pred).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (N * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)  
        sum_dim = torch.sum(pred * entropy_weight, dim = 0).unsqueeze(dim=0)
        return 1 / (self.alpha - 1) * torch.sum((1 / torch.mean(sum_dim) - torch.sum(pred ** self.alpha / sum_dim * entropy_weight, dim = -1)))


def PairEnum(x,mask=None):
    # Enumerate all pairs of feature in x
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0), 1)
    x2 = x.repeat(1, x.size(0)).view(-1, x.size(1))
    if mask is not None:
        xmask = mask.view(-1,1).repeat(1,x.size(1))
        #dim 0: #sample, dim 1:#feature 
        x1 = x1[xmask].view(-1,x.size(1))
        x2 = x2[xmask].view(-1,x.size(1))
    return x1,x2

class BCE(nn.Module):
    eps = 1e-7 # Avoid calculating log(0). Use the small value of float16.
    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1)==len(prob2)==len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),str(len(prob2)),str(len(simi)))
        P = prob1.mul_(prob2)
        P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(BCE.eps).log_()
        return neglogP.mean()

class BCEMultiTask(nn.Module):
    eps = 1e-7 # Avoid calculating log(0). Use the small value of float16.
    def forward(self, prob1, prob2, simi):
        assert len(prob1)==len(prob2)==len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),str(len(prob2)),str(len(simi)))
        n_t = prob1.shape[1]
        prob1_m = torch.zeros(prob1.shape[0], 2, n_t).to(prob1.device)
        prob2_m = torch.zeros(prob1.shape[0], 2, n_t).to(prob1.device)
        prob1_m[:, 0, :], prob1_m[:, 1, :] = prob1, 1.0 - prob1
        prob2_m[:, 0, :], prob2_m[:, 1, :] = prob2, 1.0 - prob2
        simi_m = simi.unsqueeze(1).repeat(1, n_t)
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        P = prob1_m.mul_(prob2_m)
        P = P.sum(1)
        P.mul_(simi_m).add_(simi_m.eq(-1).type_as(P))
        neglogP = -P.add_(BCE.eps).log_()
        return neglogP.mean()

class ClusterLoss():
    def __init__(self, device, num_classes, bce_type, cosine_threshold, topk, negative_thre=None):
        # super(NCLMemory, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.bce_type = bce_type
        self.costhre = cosine_threshold
        self.topk = topk
        self.negative_thre = negative_thre
        self.bce = BCE()

    def compute_losses(self, inputs, include_label=False, unlabel_only=False, matrix_only=False):
        assert (include_label == False) or (unlabel_only == False)
        bce_loss = 0.0
        device = self.device
        feat, output2 = inputs["x1"], inputs["preds1_u"]
        output2_bar = inputs["preds2_u"]
        label = inputs["labels"]

        num_s = (label < self.num_classes).sum()
        labels_s = label[:num_s]

        if unlabel_only:
            mask_lb = label < self.num_classes
        else:
            mask_lb = torch.zeros_like(label).bool()
        
        rank_feat = (feat[~mask_lb]).detach()
        if self.bce_type == 'cos':
            # default: cosine similarity with threshold
            feat_row, feat_col = PairEnum(F.normalize(rank_feat, dim=1))
            tmp_distance_ori = torch.bmm(
                feat_row.view(feat_row.size(0), 1, -1),
                feat_col.view(feat_row.size(0), -1, 1)
            )
            tmp_distance_ori = tmp_distance_ori.squeeze()
            if self.negative_thre is None:
                target_ulb = torch.zeros_like(tmp_distance_ori).float() - 1
                target_ulb[tmp_distance_ori > self.costhre] = 1
            else:
                target_ulb = torch.zeros_like(tmp_distance_ori).float()
                target_ulb[tmp_distance_ori > self.costhre] = 1
                target_ulb[tmp_distance_ori < self.negative_thre] = -1
        elif self.bce_type == 'RK':
            # top-k rank statics
            rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
            rank_idx1, rank_idx2 = PairEnum(rank_idx)
            rank_idx1, rank_idx2 = rank_idx1[:, :self.topk], rank_idx2[:, :self.topk]
            rank_idx1, _ = torch.sort(rank_idx1, dim=1)
            rank_idx2, _ = torch.sort(rank_idx2, dim=1)
            rank_diff = rank_idx1 - rank_idx2
            rank_diff = torch.sum(torch.abs(rank_diff), dim=1)
            target_ulb = torch.ones_like(rank_diff).float().to(device)
            target_ulb[rank_diff > 0] = -1

        if include_label:
            # use source domain label for similar/dissimilar
            labels = labels_s.contiguous().view(-1, 1)
            mask_l = torch.eq(labels, labels.T).float().to(device)
            mask_l = (mask_l - 0.5) * 2.0
            target_ulb_t = target_ulb.view(feat.size(0), -1)
            target_ulb_t[:num_s, :num_s] = mask_l
            target_ulb = target_ulb_t.flatten()

        if matrix_only:
            return 0.0, target_ulb
            
        prob2, prob2_bar = F.softmax(output2, dim=1), F.softmax(output2_bar, dim=1)
        prob1_ulb, _ = PairEnum(prob2[~mask_lb])
        _, prob2_ulb = PairEnum(prob2_bar[~mask_lb])

        bce_loss = self.bce(prob1_ulb, prob2_ulb, target_ulb)
        return bce_loss, target_ulb

def compute_ulb_sim_matrix(ulb_feat, bce_type, topk, costhre, device, negative_thre=None):
    rank_feat = ulb_feat.detach()
    if bce_type == 'cos':
        # default: cosine similarity with threshold
        feat_row, feat_col = PairEnum(F.normalize(rank_feat, dim=1))
        tmp_distance_ori = torch.bmm(
            feat_row.view(feat_row.size(0), 1, -1),
            feat_col.view(feat_row.size(0), -1, 1)
        )
        tmp_distance_ori = tmp_distance_ori.squeeze()
        if negative_thre is None:
            target_ulb = torch.zeros_like(tmp_distance_ori).float() - 1
            target_ulb[tmp_distance_ori > costhre] = 1
        else:
            target_ulb = torch.zeros_like(tmp_distance_ori).float()
            target_ulb[tmp_distance_ori > costhre] = 1
            target_ulb[tmp_distance_ori < negative_thre] = -1
    elif bce_type == 'RK':
        # top-k rank statics
        rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
        rank_idx1, rank_idx2 = PairEnum(rank_idx)
        rank_idx1, rank_idx2 = rank_idx1[:, :topk], rank_idx2[:, :topk]
        rank_idx1, _ = torch.sort(rank_idx1, dim=1)
        rank_idx2, _ = torch.sort(rank_idx2, dim=1)
        rank_diff = rank_idx1 - rank_idx2
        rank_diff = torch.sum(torch.abs(rank_diff), dim=1)
        target_ulb = torch.ones_like(rank_diff).float().to(device)
        target_ulb[rank_diff > 0] = -1
    return target_ulb

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        select_mask = (mask.sum(1) != 0)
        mask = mask[select_mask, :]
        log_prob = log_prob[select_mask, :]
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # filter out those with no positive samples
        mean_log_prob_pos=mean_log_prob_pos[mean_log_prob_pos==mean_log_prob_pos]  

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # loss = loss.view(anchor_count, batch_size).mean()
        loss = loss.mean()
        return loss

class DisentanglementLoss(object):
    def __init__(self, num_classes, memory_length_per_class=50, temperature=0.07, use_target_as_positive=False):
        self.memory_length = memory_length_per_class * num_classes
        self.num_classes = num_classes
        self.temperature = temperature
        self.labels_memory = None
        self.clusters_memory = None
        self.use_target_as_positive = use_target_as_positive
        self.sup_con_loss = SupConLoss(temperature=temperature)

    def __call__(self, x_s, labels_s, clusters_s, x_t, update_memory=True):
        clusters_s = clusters_s.detach().cpu().numpy()
        labels_s = labels_s.detach().cpu().numpy()
        if update_memory:
            self.push_memory(labels_s, clusters_s)
        env1_mask, env2_mask = self.get_env_masks(labels_s, clusters_s)
        # if balance_envs:
        #    env1_mask, env2_mask = self.balance_envs(env1_mask, env2_mask)
        # x_s = torch.stack(x_s.chunk(2), dim=0).permute((1, 0, 2))
        # x_t = torch.stack(x_t.chunk(2), dim=0).permute((1, 0, 2))
        sup_con_loss_env1 = self.get_supcon_loss(x_s[env1_mask, :, :], labels_s[env1_mask], x_t)
        sup_con_loss_env2 = self.get_supcon_loss(x_s[env2_mask, :, :], labels_s[env2_mask], x_t)
        irm_loss = torch.var(torch.stack([sup_con_loss_env1, sup_con_loss_env2]))
        return sup_con_loss_env1 + sup_con_loss_env2, irm_loss

    def get_regression_loss(self, x_s, labels_s, x_t, topk, margin):
        x = torch.cat((x_s, x_t), dim=0)
        num_s = x_s.size(0)
        num_st = num_s + x_t.size(0)
        mask = (compute_ulb_sim_matrix(torch.cat((x_s[:, 0, :], x_t[:, 0, :]), dim=0), "RK", topk, 0.95, x_s.device) + 1.0) / 2.0
        mask = mask.view(num_st, num_st)
        labels_s1, labels_s2 = PairEnum(labels_s)
        labels_s1, labels_s2 = labels_s1.squeeze(1), labels_s2.squeeze(1)
        mask_s = (torch.abs(labels_s1 - labels_s2) <= margin).float().view(num_s, num_s)
        mask[:num_s, :num_s] = mask_s
        return self.sup_con_loss(features=F.normalize(x, dim=-1), mask=mask), 0.0

    def push_memory(self, labels_s, clusters_s):
        clusters_s_hard = np.argmax(clusters_s, axis=1)
        if self.labels_memory is None:
            self.labels_memory = labels_s
            self.clusters_memory = clusters_s_hard
        elif len(self.labels_memory) + len(labels_s) > self.memory_length:
            self.labels_memory = np.append(self.labels_memory, labels_s)
            self.clusters_memory = np.append(self.clusters_memory, clusters_s_hard)
            self.labels_memory = self.labels_memory[-self.memory_length:]
            self.clusters_memory = self.clusters_memory[-self.memory_length:]
        else:
            self.labels_memory = np.append(self.labels_memory, labels_s)
            self.clusters_memory = np.append(self.clusters_memory, clusters_s_hard)

    def get_env_masks(self, labels_s, clusters_s):
        # dominant cluster index for each class
        dom_cluster_indices = [np.argmax(np.bincount(
            self.clusters_memory[self.labels_memory == i], minlength=self.num_classes))
            for i in range(self.num_classes)]
        dom_cluster_indices = np.array(dom_cluster_indices)

        """ # class mask: num_classes * len(x_s)
        class_mask = np.stack([(labels_s == i) for i in range(self.num_classes)])

        # cluster mask: num_classes * len(x_s)
        cluster_mask = np.stack([(clusters_s == dom_cluster_indices[i])
            for i in range(self.num_classes)]
        ) 
        # env1 idx for each class
        mask_matrix = class_mask & cluster_mask
        env1_mask = np.any(mask_matrix, axis=0) # len(x_s)
        env2_mask = ~env1_mask
        """
        scores = [clusters_s[i][dom_cluster_indices[labels_s[i]]] for i in range(len(labels_s))]
        scores = np.array(scores)

        ranks = np.argsort(scores)
        env1_mask = torch.zeros(len(labels_s)).bool()
        env1_mask[ranks[:len(labels_s) // 2]] = 1
        return env1_mask, ~env1_mask

    def balance_envs(self, env1_mask, env2_mask):
        if env1_mask.sum() > env2_mask.sum():
            larger_mask = env1_mask
        else:
            larger_mask = env2_mask
        num_selected = min(env1_mask.sum(), env2_mask.sum())
        larger_mask[larger_mask > 0][num_selected:] = 0
        return env1_mask, env2_mask

    def get_supcon_loss(self, x_s, labels_s, x_t):
        num_s = x_s.size(0)
        num_all = num_s + x_t.size(0)
        x = torch.cat((x_s, x_t), dim=0)

        # all target as negative samples
        # all target will not act as positive
        if self.use_target_as_positive:
            mask = torch.eye(num_all).float()
        else:
            mask = torch.zeros((num_all, num_all)).float()
        labels = torch.from_numpy(labels_s).contiguous().view(-1, 1)
        mask_s = torch.eq(labels, labels.T).float()
        mask[:num_s, :num_s] = mask_s
        return self.sup_con_loss(features=F.normalize(x, dim=-1), mask=mask)

sim_list = []
def get_ulb_sim_matrix(mode, sim_matrix_ulb, labels_t, cluster_preds_t, bf_t,
        device, sim_threshold=-2.0, diff_threshold=-2.0, sim_ratio=0.1, diff_ratio=0.5,
        update_list=True):
    with torch.no_grad():
        if mode == 'stats':
            p_low = float((sim_matrix_ulb < 0.0).sum()) / len(sim_matrix_ulb)
            p_high = float((sim_matrix_ulb > 0.0).sum()) / len(sim_matrix_ulb)
            return sim_matrix_ulb, 0, 0, p_low, p_high
        elif mode == 'gt':
            labels = labels_t.contiguous().view(-1, 1)
            mask_t = torch.eq(labels, labels.T).float().to(device)
            mask_t = (mask_t - 0.5) * 2.0
            return mask_t.flatten(), 0, 0, 0, 0
        else:
            # get cosine similarity
            if mode == 'sim':
                feat_row, feat_col = PairEnum(F.normalize(cluster_preds_t, dim=1))
            elif mode == 'prob':
                feat_row, feat_col = PairEnum(F.softmax(cluster_preds_t, dim=1))
            elif mode == 'fsim':
                feat_row, feat_col = PairEnum(F.normalize(bf_t, dim=1))
            tmp_distance_ori = torch.bmm(
                feat_row.view(feat_row.size(0), 1, -1),
                feat_col.view(feat_row.size(0), -1, 1)
            )
            similarity = tmp_distance_ori.squeeze()
            if update_list:
                global sim_list
                sim_list.append(similarity.cpu())
                max_len = 10 if len(similarity) > 100000 else 30
                if len(sim_list) > max_len:
                    sim_list = sim_list[1:]
            sim_all = torch.cat(sim_list, dim=0)
            sim_all_sorted, _ = torch.sort(sim_all)

            n_diff = len(sim_all) * diff_ratio
            n_sim = len(sim_all) * sim_ratio

            if abs(diff_threshold) <= 1.0:
                low_threshold = min(diff_threshold, sim_all_sorted[int(n_diff)])
            else:
                low_threshold = sim_all_sorted[int(n_diff)]

            if abs(sim_threshold) <= 1.0:
                high_threshold = max(sim_threshold, sim_all_sorted[-int(n_sim)])
            else:
                high_threshold = sim_all_sorted[-int(n_sim)]

            sim_matrix_ulb = torch.zeros_like(similarity).float()
            p_high = float((similarity >= high_threshold).sum()) / len(similarity)
            p_low = float((similarity <= low_threshold).sum()) / len(similarity)
            sim_matrix_ulb[similarity >= high_threshold] = 1.0
            sim_matrix_ulb[similarity <= low_threshold] = -1.0
            
        return sim_matrix_ulb, low_threshold, high_threshold, p_low, p_high

def calc_entropy(input_tensor):
    lsm = nn.LogSoftmax(dim=-1)
    log_probs = lsm(input_tensor)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    entropy = -p_log_p.mean()
    return entropy

onehot_list = []
def calc_diversity(cluster_logits, num_classes):
    pred_clusters = cluster_logits.detach().cpu().argmax(-1)
    onehot = F.one_hot(pred_clusters, num_classes=num_classes)
    global onehot_list
    onehot_list.append(onehot)
    if len(onehot_list) > 150:
        onehot_list = onehot_list[1:]
    onehot_global = torch.cat(onehot_list, dim=0)
    num_batch = float(onehot.shape[0])
    num_global = float(onehot_global.shape[0])
    batch_histogram = onehot.sum(dim=0).float() / num_batch + 1e-6
    global_histogram = onehot_global.sum(dim=0).float() / num_global + 1e-6
    batch_diversity = -(torch.log(batch_histogram) * batch_histogram).mean()
    global_diversity = -(torch.log(global_histogram) * global_histogram).mean()
    return batch_diversity, global_diversity