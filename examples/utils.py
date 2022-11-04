import sys
import os
import csv
import argparse
import random
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import re
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError as e:
    pass

try:
    from torch_geometric.data import Batch
except ImportError:
    pass


def cross_entropy_with_logits_loss(input, soft_target):
    """
    Implementation of CrossEntropy loss using a soft target. Extension of BCEWithLogitsLoss to MCE.
    Normally, cross entropy loss is
        \sum_j 1{j == y} -log \frac{e^{s_j}}{\sum_k e^{s_k}} = -log \frac{e^{s_y}}{\sum_k e^{s_k}}
    Here we use
        \sum_j P_j *-log \frac{e^{s_j}}{\sum_k e^{s_k}}
    where 0 <= P_j <= 1
    Does not support fancy nn.CrossEntropy options (e.g. weight, size_average, ignore_index, reductions, etc.)

    Args:
    - input (N, k): logits
    - soft_target (N, k): targets for softmax(input); likely want to use class probabilities
    Returns:
    - losses (N, 1)
    """
    return torch.sum(- soft_target * torch.nn.functional.log_softmax(input, 1), 1)

def update_average(prev_avg, prev_counts, curr_avg, curr_counts):
    denom = prev_counts + curr_counts
    if isinstance(curr_counts, torch.Tensor):
        denom += (denom==0).float()
    elif isinstance(curr_counts, int) or isinstance(curr_counts, float):
        if denom==0:
            return 0.
    else:
        raise ValueError('Type of curr_counts not recognized')
    prev_weight = prev_counts/denom
    curr_weight = curr_counts/denom
    return prev_weight*prev_avg + curr_weight*curr_avg

# Taken from https://sumit-ghosh.com/articles/parsing-dictionary-key-value-pairs-kwargs-argparse-python/
class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value_str = value.split('=')
            if value_str.replace('-','').isnumeric():
                processed_val = int(value_str)
            elif value_str.replace('-','').replace('.','').isnumeric():
                processed_val = float(value_str)
            elif value_str in ['True', 'true']:
                processed_val = True
            elif value_str in ['False', 'false']:
                processed_val = False
            else:
                processed_val = value_str
            getattr(namespace, self.dest)[key] = processed_val

def parse_bool(v):
    if v.lower()=='true':
        return True
    elif v.lower()=='false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def save_model(algorithm, epoch, best_val_metric, path):
    state = {}
    state['algorithm'] = algorithm.state_dict()
    state['epoch'] = epoch
    state['best_val_metric'] = best_val_metric
    torch.save(state, path)

def load(module, path, device=None, tries=2):
    """
    Handles loading weights saved from this repo/model into an algorithm/model.
    Attempts to handle key mismatches between this module's state_dict and the loaded state_dict.
    Args:
        - module (torch module): module to load parameters for
        - path (str): path to .pth file
        - device: device to load tensors on
        - tries: number of times to run the match_keys() function
    """
    if device is not None:
        state = torch.load(path, map_location=device)
    else:
        state = torch.load(path)

    # Loading from a saved WILDS Algorithm object
    if 'algorithm' in state:
        prev_epoch = state['epoch']
        best_val_metric = state['best_val_metric']
        state = state['algorithm']
    # Loading from a pretrained SwAV model
    elif 'state_dict' in state:
        state = state['state_dict']
        prev_epoch, best_val_metric = None, None
    else:
        prev_epoch, best_val_metric = None, None

    # If keys match perfectly, load_state_dict() will work
    try: module.load_state_dict(state)
    except:
        # Otherwise, attempt to reconcile mismatched keys and load with strict=False
        module_keys = module.state_dict().keys()
        for _ in range(tries):
            state = match_keys(state, list(module_keys))
            module.load_state_dict(state, strict=False)
            leftover_state = {k:v for k,v in state.items() if k in list(state.keys()-module_keys)}
            leftover_module_keys = module_keys - state.keys()
            if len(leftover_state) == 0 or len(leftover_module_keys) == 0: break
            state, module_keys = leftover_state, leftover_module_keys
        if len(module_keys-state.keys()) > 0: print(f"Some module parameters could not be found in the loaded state: {module_keys-state.keys()}")
    return prev_epoch, best_val_metric

def match_keys(d, ref):
    """
    Matches the format of keys between d (a dict) and ref (a list of keys).

    Helper function for situations where two algorithms share the same model, and we'd like to warm-start one
    algorithm with the model of another. Some algorithms (e.g. FixMatch) save the featurizer, classifier within a sequential,
    and thus the featurizer keys may look like 'model.module.0._' 'model.0._' or 'model.module.model.0._',
    and the classifier keys may look like 'model.module.1._' 'model.1._' or 'model.module.model.1._'
    while simple algorithms (e.g. ERM) use no sequential 'model._'
    """
    # hard-coded exceptions
    d = {re.sub('model.1.', 'model.classifier.', k): v for k,v in d.items()}
    d = {k: v for k,v in d.items() if 'pre_classifier' not in k} # this causes errors

    # probe the proper transformation from d.keys() -> reference
    # do this by splitting d's first key on '.' until we get a string that is a strict substring of something in ref
    success = False
    probe = list(d.keys())[0].split('.')
    for i in range(len(probe)):
        probe_str = '.'.join(probe[i:])
        matches = list(filter(lambda ref_k: len(ref_k) >= len(probe_str) and probe_str == ref_k[-len(probe_str):], ref))
        matches = list(filter(lambda ref_k: not 'layer' in ref_k, matches)) # handle resnet probe being too simple, e.g. 'weight'
        if len(matches) == 0: continue
        else:
            success = True
            append = [m[:-len(probe_str)] for m in matches]
            remove = '.'.join(probe[:i]) + '.'
            break
    if not success: raise Exception("These dictionaries have irreconcilable keys")

    return_d = {}
    for a in append:
        for k,v in d.items(): return_d[re.sub(remove, a, k)] = v

    # hard-coded exceptions
    if 'model.classifier.weight' in return_d:
       return_d['model.1.weight'], return_d['model.1.bias'] = return_d['model.classifier.weight'], return_d['model.classifier.bias']
    return return_d

def log_group_data(datasets, grouper, logger):
    for k, dataset in datasets.items():
        name = dataset['name']
        dataset = dataset['dataset']
        logger.write(f'{name} data...\n')
        if grouper is None:
            logger.write(f'    n = {len(dataset)}\n')
        else:
            _, group_counts = grouper.metadata_to_group(
                dataset.metadata_array,
                return_counts=True)
            group_counts = group_counts.tolist()
            for group_idx in range(grouper.n_groups):
                logger.write(f'    {grouper.group_str(group_idx)}: n = {group_counts[group_idx]:.0f}\n')
    logger.flush()

class Logger(object):
    def __init__(self, fpath=None, mode='w'):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, mode)

    def change_fpath(self, fpath, mode='w'):
        if self.file is not None:
            self.file.close()
        self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

class BatchLogger:
    def __init__(self, csv_path, mode='w', use_wandb=False):
        self.path = csv_path
        self.mode = mode
        self.file = open(csv_path, mode)
        self.is_initialized = False

        # Use Weights and Biases for logging
        self.use_wandb = use_wandb
        if use_wandb:
            self.split = Path(csv_path).stem

    def setup(self, log_dict):
        columns = log_dict.keys()
        # Move epoch and batch to the front if in the log_dict
        for key in ['batch', 'epoch']:
            if key in columns:
                columns = [key] + [k for k in columns if k != key]

        self.writer = csv.DictWriter(self.file, fieldnames=columns)
        if self.mode=='w' or (not os.path.exists(self.path)) or os.path.getsize(self.path)==0:
            self.writer.writeheader()
        self.is_initialized = True

    def log(self, log_dict):
        if self.is_initialized is False:
            self.setup(log_dict)
        self.writer.writerow(log_dict)
        self.flush()

        if self.use_wandb:
            results = {}
            for key in log_dict:
                new_key = f'{self.split}/{key}'
                results[new_key] = log_dict[key]
            wandb.log(results)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()

def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def log_config(config, logger):
    for name, val in vars(config).items():
        logger.write(f'{name.replace("_"," ").capitalize()}: {val}\n')
    logger.write('\n')

def initialize_wandb(config):
    if config.wandb_api_key_path is not None:
        with open(config.wandb_api_key_path, "r") as f:
            os.environ["WANDB_API_KEY"] = f.read().strip()

    wandb.init(**config.wandb_kwargs)
    wandb.config.update(config)

def save_pred(y_pred, path_prefix):
    # Single tensor
    if torch.is_tensor(y_pred):
        df = pd.DataFrame(y_pred.numpy())
        df.to_csv(path_prefix + '.csv', index=False, header=False)
    # Dictionary
    elif isinstance(y_pred, dict) or isinstance(y_pred, list):
        torch.save(y_pred, path_prefix + '.pth')
    else:
        raise TypeError("Invalid type for save_pred")

def get_replicate_str(dataset, config):
    if dataset['dataset'].dataset_name == 'poverty':
        replicate_str = f"fold:{config.dataset_kwargs['fold']}"
    else:
        replicate_str = f"seed:{config.seed}"
    return replicate_str

def get_pred_prefix(dataset, config):
    dataset_name = dataset['dataset'].dataset_name
    split = dataset['split']
    replicate_str = get_replicate_str(dataset, config)
    prefix = os.path.join(
        config.log_dir,
        f"{dataset_name}_split:{split}_{replicate_str}_")
    return prefix

def get_model_prefix(dataset, config):
    dataset_name = dataset['dataset'].dataset_name
    replicate_str = get_replicate_str(dataset, config)
    prefix = os.path.join(
        config.log_dir,
        f"{dataset_name}_{replicate_str}_")
    return prefix

def move_to(obj, device):
    if isinstance(obj, dict):
        return {k: move_to(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to(v, device) for v in obj]
    elif isinstance(obj, float) or isinstance(obj, int):
        return obj
    else:
        # Assume obj is a Tensor or other type
        # (like Batch, for MolPCBA) that supports .to(device)
        return obj.to(device)

def detach_and_clone(obj):
    if torch.is_tensor(obj):
        return obj.detach().clone()
    elif isinstance(obj, dict):
        return {k: detach_and_clone(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [detach_and_clone(v) for v in obj]
    elif isinstance(obj, float) or isinstance(obj, int):
        return obj
    else:
        raise TypeError("Invalid type for detach_and_clone")

def collate_list(vec):
    """
    If vec is a list of Tensors, it concatenates them all along the first dimension.

    If vec is a list of lists, it joins these lists together, but does not attempt to
    recursively collate. This allows each element of the list to be, e.g., its own dict.

    If vec is a list of dicts (with the same keys in each dict), it returns a single dict
    with the same keys. For each key, it recursively collates all entries in the list.
    """
    if not isinstance(vec, list):
        raise TypeError("collate_list must take in a list")
    elem = vec[0]
    if torch.is_tensor(elem):
        return torch.cat(vec)
    elif isinstance(elem, list):
        return [obj for sublist in vec for obj in sublist]
    elif isinstance(elem, dict):
        return {k: collate_list([d[k] for d in vec]) for k in elem}
    else:
        raise TypeError("Elements of the list to collate must be tensors or dicts.")

def remove_key(key):
    """
    Returns a function that strips out a key from a dict.
    """
    def remove(d):
        if not isinstance(d, dict):
            raise TypeError("remove_key must take in a dict")
        return {k: v for (k,v) in d.items() if k != key}
    return remove

def concat_input(labeled_x, unlabeled_x):
    if isinstance(labeled_x, torch.Tensor):
        x_cat = torch.cat((labeled_x, unlabeled_x), dim=0)
    elif isinstance(labeled_x, Batch):
        labeled_x.y = None
        x_cat = Batch.from_data_list([labeled_x, unlabeled_x])
    else:
        raise TypeError("x must be Tensor or Batch")
    return x_cat

def update_regd_log_dir(config):
    args = config

    if args.dataset != 'ogb-molpcba':
        exp_name = "%s" % (args.dataset)
    else:
        exp_name = "ogb"
        
    if args.exp_name != "":
        if args.dataset == 'poverty':
            exp_name += "_%s-s%s" % (args.exp_name, args.dataset_kwargs['fold'])
        else:
            exp_name += "_%s-s%d" % (args.exp_name, args.seed)

    exp_name += "_b%d-n%d" % (args.batch_size, args.n_epochs)
    if args.erm_start_epoch != 0:
        exp_name += "_erm%d" % args.erm_start_epoch
    exp_name += "_lr%.5f_bf%d_dp%.1f" % (args.lr, args.bottleneck_dim, args.regd_dropout)

    if args.noisy_student:
        exp_name += "_ns%d-%d-%.2f" % (args.regd_ps_freq, args.consistency_start_epoch, args.w_con)
    else:
        exp_name += "_fm-%d-%.2f-%.1f" % (args.consistency_start_epoch,
            args.self_training_threshold, args.w_con)
        if args.pseudolabel_T2 != 0.4:
            exp_name += "-T%.1f" % args.pseudolabel_T2

    if args.bottleneck_cluster:
        exp_name += "_bc"
        
    exp_name += "_cl%.1f" % (args.w_cluster)
    if args.bce_type == 'RK':
        exp_name += "-rk%d" % args.topk
    else:
        exp_name += "-cos%.2f-%.2f" % (args.cosine_threshold, args.bce_diff_threshold)
    
    if args.back_cluster:
        exp_name += "_ba%d" % args.back_cluster_start_epoch

    exp_name += "_irm%d-%.1f-%.1f" % (args.irm_start_epoch, args.w_irm, args.w_supcon)
    if args.dataset == 'poverty':
        exp_name += "-%.2f" % args.supcon_margin

    exp_name += "_tr%.3f-%.1f-%.1f" % (args.w_transfer, args.ts_temperature, args.ts_alpha)
    exp_name += "_temp%.3f" % args.supcon_temperature


    exp_name += "_compat-%s-%.2f-%d" % (args.compat_mode, args.w_compat, args.compat_start_epoch)
    if args.w_compat_u != 1.0:
        exp_name += "-%.1f" % args.w_compat_u
    if args.compat_mode == "sim" or args.compat_mode == "prob":
        exp_name += "-s%.2f-%.2f-%.2f_d%.2f-%.2f-%.2f" % (args.compat_sim_threshold, args.compat_sim_ratio,
                args.compat_sim_ratio_delta, args.compat_diff_threshold, args.compat_diff_ratio, args.compat_diff_ratio_delta)
        if args.dataset == 'ogb-molpcba' or args.dataset == 'poverty':
            exp_name += "-%d" % args.num_pseudo_classes
    if args.dataset == 'poverty':
        exp_name += "-m%.2f-s%.2f" % (args.compat_margin, args.compat_sim_margin)
    if args.back_compat_l:
        exp_name += "-l"
    if args.back_compat_u:
        exp_name += "-u"
    if args.dataset == 'ogb-molpcba':
        if args.compat_multitask_normalize:
            exp_name += "-nom"
        exp_name += "-%.1f" % (args.compat_multitask_margin)

    if args.w_compat_irm != 0.0:
        exp_name += "_%s%.2f-s%d" % (args.compat_irm_sim_mode, args.w_compat_irm, args.compat_irm_start_epoch)
        if args.back_compat_irm:
            exp_name += "-b"

    if args.dim_reduction != 'none':
        exp_name += "%s-%d" % (args.dim_reduction, args.reduced_dim)
        
    return os.path.join(config.log_dir, exp_name)

class InfiniteDataIterator:
    """
    Adapted from https://github.com/thuml/Transfer-Learning-Library

    A data iterator that will never stop producing data
    """
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            print("Reached the end, resetting data loader...")
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)

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

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Visualizer():
    def __init__(self, log_dir):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir)

    def plot_items(self, iteration, items):
        for name, value in items.items():
            self.writer.add_scalar(name, value, iteration)

def print_train_info(train_info_list, config=None):
    n_epochs = len(train_info_list[0]['test'])
    n_seeds = len(train_info_list)
    test_best_matrix = np.zeros((n_epochs + 1, n_seeds))
    # Compute test best
    for i in range(n_epochs):
        for seed in range(n_seeds):
            if train_info_list[seed]['test'][i] > test_best_matrix[i][seed]:
                test_best_matrix[i + 1][seed] = train_info_list[seed]['test'][i]
            else:
                test_best_matrix[i + 1][seed] = test_best_matrix[i][seed]
    best_epoch_acc = test_best_matrix.max(axis=0)
    print("Test best:")
    print(best_epoch_acc)
    print("Mean: %.4f, Std: %.4f" % (np.mean(best_epoch_acc), np.std(best_epoch_acc)))

    with open(log_file, "a") as file:
        if config is not None:
            for name, val in vars(config).items():
                file.write(f'{name.replace("_"," ").capitalize()}: {val}, ')
            file.write('\n')
        file.write("Test best -- Mean: %.4f, Std: %.4f\n" % (np.mean(best_epoch_acc), np.std(best_epoch_acc)))

    # Compute val best
    val_best = np.zeros(n_seeds)
    val_best_matrix = np.zeros((n_epochs, n_seeds))
    for i in range(n_epochs):
        for seed in range(n_seeds):
            if train_info_list[seed]['val'][i] > val_best[seed]:
                val_best[seed] = train_info_list[seed]['val'][i]
                val_best_matrix[i][seed] = train_info_list[seed]['test'][i]
            else:
                val_best_matrix[i][seed] = val_best_matrix[i - 1][seed]
    mean_val_best = np.mean(val_best_matrix, axis=1)
    best_epoch = np.argmax(mean_val_best)
    best_epoch_acc = val_best_matrix[best_epoch]
    print("Val best:")
    print(best_epoch_acc)
    print("Epoch: %d, Mean: %.4f, Std: %.4f" % (best_epoch, np.max(mean_val_best), np.std(best_epoch_acc)))
    
    with open(log_file, "a") as file:
        file.write("Val best -- Epoch: %d, Mean: %.4f, Std: %.4f\n" % (best_epoch, np.max(mean_val_best), np.std(best_epoch_acc)))

    # Compute last best
    last_best_matrix = np.zeros((n_epochs, n_seeds))
    for i in range(n_epochs):
        for seed in range(n_seeds):
            last_best_matrix[i][seed] = train_info_list[seed]['test'][i]
    mean_last_best = np.mean(last_best_matrix, axis=1)
    best_epoch = np.argmax(mean_last_best)
    best_epoch_acc = last_best_matrix[best_epoch]
    print("Last best:")
    print(best_epoch_acc)
    print("Epoch: %d, Mean: %.4f, Std: %.4f" % (best_epoch, np.max(mean_last_best), np.std(best_epoch_acc)))

    with open(log_file, "a") as file:
        file.write("Last best -- Epoch: %d, Mean: %.4f, Std: %.4f\n\n" % (best_epoch, np.max(mean_last_best), np.std(best_epoch_acc)))


def collect_feature(data_loader, feature_extractor, device, feat_dim, two_views, has_labels=False):
    """
    Fetch data from `data_loader`, and then use `feature_extractor` to collect features

    Args:
        data_loader (torch.utils.data.DataLoader): Data loader.
        feature_extractor (torch.nn.Module): A feature extractor.
        device (torch.device)
        max_num_features (int): The max number of features to return

    Returns:
        Features in shape (min(len(data_loader), max_num_features), :math:`|\mathcal{F}|`).
    """
    feature_extractor.eval()
    N = len(data_loader.dataset)
    all_features = torch.zeros(N, feat_dim)
    labels = []
    import tqdm
    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(data_loader)):
            if two_views:
                images = batch[0][0].to(device)
            else:
                images = batch[0].to(device)
            if has_labels:
                labels.append(batch[1])
            metadata = batch[-1]
            feature = feature_extractor(images).cpu()
            idx = metadata[:, -1].long()
            # feature2 = feature_extractor(images2).cpu()
            all_features[idx] = feature
            # all_features2.append(feature2)
            #all_labels.append(target)
    if has_labels:
        return all_features, torch.cat(labels, dim=0)
    else:
        return all_features

def reduce_dimension(features, mode, dim):
    if mode == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=dim)
        transformed_features = pca.fit_transform(features)
        fit_score = pca.explained_variance_ratio_.sum()
    elif mode == 'umap':
        import umap
        fit = umap.UMAP(n_components=dim)
        transformed_features = fit.fit_transform(features)
        fit_score = 0.0
    return transformed_features, fit_score