import os
import time
import argparse
import xxlimited
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import sys
from torch.nn import DataParallel
from collections import defaultdict
import pickle
import numpy as np
try:
    import wandb
except Exception as e:
    pass

import wilds
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.grouper import CombinatorialGrouper
from wilds.datasets.unlabeled.wilds_unlabeled_dataset import WILDSPseudolabeledSubset

from utils import set_seed, Logger, BatchLogger, log_config, ParseKwargs, load, initialize_wandb, log_group_data, parse_bool, get_model_prefix, move_to, update_regd_log_dir, print_train_info
from train import train, evaluate, infer_predictions, learn_pcg_unlabel_head
from algorithms.initializer import initialize_algorithm, infer_d_out
from transforms import initialize_transform
from models.initializer import initialize_model
from configs.utils import populate_defaults
import configs.supported as supported

import torch.multiprocessing

# Necessary for large images of GlobalWheat
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main(config, logger):
    if config.gpu >= 0:
        torch.cuda.set_device(config.gpu)
        
    # Initialize logs
    if "RegD" in config.algorithm:
        config.log_dir = update_regd_log_dir(config)
    if os.path.exists(config.log_dir) and config.resume:
        resume=True
        mode='a'
    elif os.path.exists(config.log_dir) and config.eval_only:
        resume=False
        mode='a'
    else:
        resume=False
        mode='w'

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    # logger = Logger(os.path.join(config.log_dir, 'log.txt'), mode)
    logger.change_fpath(os.path.join(config.log_dir, 'log.txt'), mode)

    # Record config
    log_config(config, logger)

    # Set random seed
    set_seed(config.seed)

    # Data
    full_dataset = wilds.get_dataset(
        dataset=config.dataset,
        version=config.version,
        root_dir=config.root_dir,
        download=config.download,
        split_scheme=config.split_scheme,
        **config.dataset_kwargs)

    # Transforms & data augmentations for labeled dataset
    # To modify data augmentation, modify the following code block.
    # If you want to use transforms that modify both `x` and `y`,
    # set `do_transform_y` to True when initializing the `WILDSSubset` below.
    if config.algorithm == "PCG":
        if config.pcg_super_contra:
            # if supervised contrastive, then provides weak, strong view
            config.additional_train_transform = "fixmatch"

    print("Using %s train transform." % config.additional_train_transform)     
    train_transform = initialize_transform(
        transform_name=config.transform,
        config=config,
        dataset=full_dataset,
        additional_transform_name=config.additional_train_transform,
        is_training=True)
    eval_transform = initialize_transform(
        transform_name=config.transform,
        config=config,
        dataset=full_dataset,
        is_training=False)

    # Configure unlabeled datasets
    unlabeled_dataset = None
    if config.unlabeled_split is not None:
        split = config.unlabeled_split
        full_unlabeled_dataset = wilds.get_dataset(
            dataset=config.dataset,
            version=config.unlabeled_version,
            root_dir=config.root_dir,
            download=config.download,
            unlabeled=True,
            **config.dataset_kwargs
        )
        train_grouper = CombinatorialGrouper(
            dataset=[full_dataset, full_unlabeled_dataset],
            groupby_fields=config.groupby_fields
        )

        # Transforms & data augmentations for unlabeled dataset
        if config.algorithm == "FixMatch" or config.algorithm == "PCG":
            # For FixMatch, we need our loader to return batches in the form ((x_weak, x_strong), m)
            # We do this by initializing a special transform function
            unlabeled_train_transform = initialize_transform(
                config.transform, config, full_dataset, is_training=True, additional_transform_name="fixmatch"
            )
        else:
            # Otherwise, use the same data augmentations as the labeled data.
            unlabeled_train_transform = train_transform

        if config.algorithm == "NoisyStudent":
            # For Noisy Student, we need to first generate pseudolabels using the teacher
            # and then prep the unlabeled dataset to return these pseudolabels in __getitem__
            print("Inferring teacher pseudolabels for Noisy Student")
            assert config.teacher_model_path is not None
            if not config.teacher_model_path.endswith(".pth"):
                # Use the best model
                config.teacher_model_path = os.path.join(
                    config.teacher_model_path,  f"{config.dataset}_seed:{config.seed}_epoch:best_model.pth"
                )

            d_out = infer_d_out(full_dataset, config)
            teacher_model = initialize_model(config, d_out).to(config.device)
            load(teacher_model, config.teacher_model_path, device=config.device)
            # Infer teacher outputs on weakly augmented unlabeled examples in sequential order
            weak_transform = initialize_transform(
                transform_name=config.transform,
                config=config,
                dataset=full_dataset,
                is_training=True,
                additional_transform_name="weak"
            )
            unlabeled_split_dataset = full_unlabeled_dataset.get_subset(split, transform=weak_transform, frac=config.frac)
            sequential_loader = get_eval_loader(
                loader=config.eval_loader,
                dataset=unlabeled_split_dataset,
                grouper=train_grouper,
                batch_size=config.unlabeled_batch_size,
                **config.unlabeled_loader_kwargs
            )
            teacher_outputs = infer_predictions(teacher_model, sequential_loader, config)
            teacher_outputs = move_to(teacher_outputs, torch.device("cpu"))
            unlabeled_split_dataset = WILDSPseudolabeledSubset(
                reference_subset=unlabeled_split_dataset,
                pseudolabels=teacher_outputs,
                transform=unlabeled_train_transform,
                collate=full_dataset.collate,
            )
            teacher_model = teacher_model.to(torch.device("cpu"))
            del teacher_model
        elif config.algorithm == "RegD":
            # Create sequential loader to evaluate pseudo labels
            weak_transform = initialize_transform(
                transform_name=config.transform,
                config=config,
                dataset=full_dataset,
                is_training=True,
                additional_transform_name="weak"
            )
            weak_unlabeled_split_dataset = full_unlabeled_dataset.get_subset(split, transform=weak_transform, frac=config.frac)
            sequential_loader = get_eval_loader(
                loader=config.eval_loader,
                dataset=weak_unlabeled_split_dataset,
                grouper=train_grouper,
                batch_size=config.unlabeled_batch_size,
                **config.unlabeled_loader_kwargs
            )
            # Create unlabel split dataset to sample unlabel batches
            if config.dataset == 'poverty':
                dummy_pseodus = torch.zeros(len(weak_unlabeled_split_dataset)).float()
            else:
                dummy_pseodus = torch.zeros(len(weak_unlabeled_split_dataset)).long()
            dummy_pseodus = move_to(dummy_pseodus, torch.device("cpu"))
            unlabeled_split_dataset = WILDSPseudolabeledSubset(
                reference_subset=weak_unlabeled_split_dataset,
                pseudolabels=dummy_pseodus,
                transform=unlabeled_train_transform,
                collate=full_dataset.collate,
            )
        elif config.algorithm in ["RegDNA", "RegDet"]:
            # Create sequential loader to evaluate pseudo labels
            weak_transform = initialize_transform(
                transform_name=config.transform,
                config=config,
                dataset=full_dataset,
                is_training=True,
                additional_transform_name="weak"
            )
            weak_unlabeled_split_dataset = full_unlabeled_dataset.get_subset(split, transform=weak_transform, frac=config.frac)
            sequential_loader = get_eval_loader(
                loader=config.eval_loader,
                dataset=weak_unlabeled_split_dataset,
                grouper=train_grouper,
                batch_size=config.unlabeled_batch_size,
                **config.unlabeled_loader_kwargs
            )
            if config.noisy_student:
                dummy_pseodus = torch.zeros(len(weak_unlabeled_split_dataset)).long()
                dummy_pseodus = move_to(dummy_pseodus, torch.device("cpu"))
                unlabeled_split_dataset = WILDSPseudolabeledSubset(
                    reference_subset=weak_unlabeled_split_dataset,
                    pseudolabels=dummy_pseodus,
                    transform=unlabeled_train_transform,
                    collate=full_dataset.collate,
                )
            else:
                unlabeled_split_dataset = full_unlabeled_dataset.get_subset(
                    split, 
                    transform=unlabeled_train_transform, 
                    frac=config.frac, 
                    load_y=config.use_unlabeled_y
                )
        elif config.algorithm == "PCG":
            assert config.teacher_model_path is not None
            d_out = infer_d_out(full_dataset, config)
            teacher_featurizer, teacher_classifier = initialize_model(config, d_out, is_featurizer=True)
            featurizer_d_out = teacher_featurizer.d_out
            if config.use_data_parallel:
                teacher_featurizer = DataParallel(teacher_featurizer)
            teacher_featurizer = teacher_featurizer.to(config.device)
            load(teacher_featurizer, config.teacher_model_path, device=config.device)
            two_view_weak_transform = initialize_transform(
                transform_name=config.transform,
                config=config,
                dataset=full_dataset,
                is_training=True,
                additional_transform_name="pcg"
            )
            pcg_transform_unlabel_dataset = full_unlabeled_dataset.get_subset(
                split,
                transform=two_view_weak_transform, 
                frac=config.frac, 
                load_y=config.use_unlabeled_y
            )
            unlabeled_split_dataset_seq = full_unlabeled_dataset.get_subset(
                split,
                transform=unlabeled_train_transform, 
                frac=config.frac, 
                load_y=config.use_unlabeled_y
            )
            pcg_head_loader = get_train_loader(
                loader=config.train_loader,
                dataset=pcg_transform_unlabel_dataset,
                batch_size=config.pcg_unlabel_bs,
                uniform_over_groups=config.uniform_over_groups,
                grouper=train_grouper,
                distinct_groups=config.distinct_groups,
                n_groups_per_batch=config.unlabeled_n_groups_per_batch,
                **config.unlabeled_loader_kwargs
            )
            sequential_loader = get_eval_loader(
                loader=config.eval_loader,
                dataset=unlabeled_split_dataset_seq,
                grouper=train_grouper,
                batch_size=config.pcg_unlabel_bs,
                **config.unlabeled_loader_kwargs
            )
            teacher_outputs = torch.zeros(len(unlabeled_split_dataset_seq))
            teacher_outputs = move_to(teacher_outputs, torch.device("cpu"))
            unlabeled_split_dataset = WILDSPseudolabeledSubset(
                reference_subset=unlabeled_split_dataset_seq,
                pseudolabels=teacher_outputs,
                transform=unlabeled_train_transform,
                collate=full_dataset.collate,
            )
        else:
            unlabeled_split_dataset = full_unlabeled_dataset.get_subset(
                split, 
                transform=unlabeled_train_transform, 
                frac=config.frac, 
                load_y=config.use_unlabeled_y
            )

        unlabeled_dataset = {
            'split': split,
            'name': full_unlabeled_dataset.split_names[split],
            'dataset': unlabeled_split_dataset
        }
        unlabeled_dataset['loader'] = get_train_loader(
            loader=config.train_loader,
            dataset=unlabeled_dataset['dataset'],
            batch_size=config.unlabeled_batch_size,
            uniform_over_groups=config.uniform_over_groups,
            grouper=train_grouper,
            distinct_groups=config.distinct_groups,
            n_groups_per_batch=config.unlabeled_n_groups_per_batch,
            **config.unlabeled_loader_kwargs
        )
    else:
        train_grouper = CombinatorialGrouper(
            dataset=full_dataset,
            groupby_fields=config.groupby_fields
        )
        # config.pcg_unlabel_head = None

    # Configure labeled torch datasets (WILDS dataset splits)
    datasets = defaultdict(dict)
    for split in full_dataset.split_dict.keys():
        if split=='train':
            transform = train_transform
            verbose = True
        elif split == 'val':
            transform = eval_transform
            verbose = True
        else:
            transform = eval_transform
            verbose = False
        # Get subset
        datasets[split]['dataset'] = full_dataset.get_subset(
            split,
            frac=config.frac,
            transform=transform)

        if split == 'train':
            datasets[split]['loader'] = get_train_loader(
                loader=config.train_loader,
                dataset=datasets[split]['dataset'],
                batch_size=config.batch_size,
                uniform_over_groups=config.uniform_over_groups,
                grouper=train_grouper,
                distinct_groups=config.distinct_groups,
                n_groups_per_batch=config.n_groups_per_batch,
                **config.loader_kwargs)
        else:
            datasets[split]['loader'] = get_eval_loader(
                loader=config.eval_loader,
                dataset=datasets[split]['dataset'],
                grouper=train_grouper,
                batch_size=config.batch_size,
                **config.loader_kwargs)

        # Set fields
        datasets[split]['split'] = split
        datasets[split]['name'] = full_dataset.split_names[split]
        datasets[split]['verbose'] = verbose

        # Loggers
        datasets[split]['eval_logger'] = BatchLogger(
            os.path.join(config.log_dir, f'{split}_eval.csv'), mode=mode, use_wandb=config.use_wandb
        )
        datasets[split]['algo_logger'] = BatchLogger(
            os.path.join(config.log_dir, f'{split}_algo.csv'), mode=mode, use_wandb=config.use_wandb
        )

    if config.use_wandb:
        initialize_wandb(config)

    # Logging dataset info
    # Show class breakdown if feasible
    if config.no_group_logging and full_dataset.is_classification and full_dataset.y_size==1 and full_dataset.n_classes <= 10:
        log_grouper = CombinatorialGrouper(
            dataset=full_dataset,
            groupby_fields=['y'])
    elif config.no_group_logging:
        log_grouper = None
    else:
        log_grouper = train_grouper
    log_group_data(datasets, log_grouper, logger)
    if unlabeled_dataset is not None:
        log_group_data({"unlabeled": unlabeled_dataset}, log_grouper, logger)

    # Initialize algorithm & load pretrained weights if provided
    algorithm = initialize_algorithm(
        config=config,
        datasets=datasets,
        train_grouper=train_grouper,
        unlabeled_dataset=unlabeled_dataset,
    )
    
    if config.algorithm == "PCG":
        algorithm.two_view_loader = pcg_head_loader
        algorithm.seq_loader = sequential_loader
        algorithm.unlabel_dataset = unlabeled_split_dataset
    elif "RegD" in config.algorithm:
        algorithm.seq_loader = sequential_loader
        algorithm.unlabel_dataset = unlabeled_split_dataset
        algorithm.source_loader = get_eval_loader(
            loader=config.eval_loader,
            dataset=datasets['train']['dataset'],
            grouper=train_grouper,
            batch_size=config.batch_size,
            **config.loader_kwargs
        )

    model_prefix = get_model_prefix(datasets['train'], config)
    if not config.eval_only:
        # Resume from most recent model in log_dir
        resume_success = False
        if resume:
            save_path = model_prefix + 'epoch:last_model.pth'
            if not os.path.exists(save_path):
                epochs = [
                    int(file.split('epoch:')[1].split('_')[0])
                    for file in os.listdir(config.log_dir) if file.endswith('.pth')]
                if len(epochs) > 0:
                    latest_epoch = max(epochs)
                    save_path = model_prefix + f'epoch:{latest_epoch}_model.pth'
            try:
                prev_epoch, best_val_metric = load(algorithm, save_path, device=config.device)
                epoch_offset = prev_epoch + 1
                logger.write(f'Resuming from epoch {epoch_offset} with best val metric {best_val_metric}')
                resume_success = True
            except FileNotFoundError:
                pass
        if resume_success == False:
            epoch_offset=0
            best_val_metric=None

        # Log effective batch size
        if config.gradient_accumulation_steps > 1:
            logger.write(
                (f'\nUsing gradient_accumulation_steps {config.gradient_accumulation_steps} means that')
                + (f' the effective labeled batch size is {config.batch_size * config.gradient_accumulation_steps}')
                + (f' and the effective unlabeled batch size is {config.unlabeled_batch_size * config.gradient_accumulation_steps}' 
                    if unlabeled_dataset and config.unlabeled_batch_size else '')
                + ('. Updates behave as if torch loaders have drop_last=False\n')
            )

        train_info = train(
            algorithm=algorithm,
            datasets=datasets,
            general_logger=logger,
            config=config,
            epoch_offset=epoch_offset,
            best_val_metric=best_val_metric,
            unlabeled_dataset=unlabeled_dataset,
        )
    else:
        if config.eval_epoch is None:
            eval_model_path = model_prefix + 'epoch:best_model.pth'
        else:
            eval_model_path = model_prefix +  f'epoch:{config.eval_epoch}_model.pth'
        best_epoch, best_val_metric = load(algorithm, eval_model_path, device=config.device)
        if config.eval_epoch is None:
            epoch = best_epoch
        else:
            epoch = config.eval_epoch
        if epoch == best_epoch:
            is_best = True
        evaluate(
            algorithm=algorithm,
            datasets=datasets,
            epoch=epoch,
            general_logger=logger,
            config=config,
            is_best=is_best)
        train_info = None

    if config.use_wandb:
        wandb.finish()
    
    for split in datasets:
        datasets[split]['eval_logger'].close()
        datasets[split]['algo_logger'].close()

    return train_info

if __name__=='__main__':
    ''' Arg defaults are filled in according to examples/configs/ '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('-d', '--dataset', choices=wilds.supported_datasets, required=True)
    parser.add_argument('--algorithm', required=True, choices=supported.algorithms)
    parser.add_argument('--root_dir', required=True,
                        help='The directory where [dataset]/data can be found (or should be downloaded to, if it does not exist).')

    # Dataset
    parser.add_argument('--split_scheme', help='Identifies how the train/val/test split is constructed. Choices are dataset-specific.')
    parser.add_argument('--dataset_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for dataset initialization passed as key1=value1 key2=value2')
    parser.add_argument('--download', default=False, type=parse_bool, const=True, nargs='?',
                        help='If true, tries to download the dataset if it does not exist in root_dir.')
    parser.add_argument('--frac', type=float, default=1.0,
                        help='Convenience parameter that scales all dataset splits down to the specified fraction, for development purposes. Note that this also scales the test set down, so the reported numbers are not comparable with the full test set.')
    parser.add_argument('--version', default=None, type=str, help='WILDS labeled dataset version number.')

    # Unlabeled Dataset
    parser.add_argument('--unlabeled_split', default=None, type=str, choices=wilds.unlabeled_splits,  help='Unlabeled split to use. Some datasets only have some splits available.')
    parser.add_argument('--unlabeled_version', default=None, type=str, help='WILDS unlabeled dataset version number.')
    parser.add_argument('--use_unlabeled_y', default=False, type=parse_bool, const=True, nargs='?', 
                        help='If true, unlabeled loaders will also the true labels for the unlabeled data. This is only available for some datasets. Used for "fully-labeled ERM experiments" in the paper. Correct functionality relies on CrossEntropyLoss using ignore_index=-100.')

    # Loaders
    parser.add_argument('--loader_kwargs', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--unlabeled_loader_kwargs', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--train_loader', choices=['standard', 'group'])
    parser.add_argument('--uniform_over_groups', type=parse_bool, const=True, nargs='?', help='If true, sample examples such that batches are uniform over groups.')
    parser.add_argument('--distinct_groups', type=parse_bool, const=True, nargs='?', help='If true, enforce groups sampled per batch are distinct.')
    parser.add_argument('--n_groups_per_batch', type=int)
    parser.add_argument('--unlabeled_n_groups_per_batch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--unlabeled_batch_size', type=int)
    parser.add_argument('--eval_loader', choices=['standard'], default='standard')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of batches to process before stepping optimizer and schedulers. If > 1, we simulate having a larger effective batch size (though batchnorm behaves differently).')

    # Model
    parser.add_argument('--model', choices=supported.models)
    parser.add_argument('--model_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for model initialization passed as key1=value1 key2=value2')
    parser.add_argument('--noisystudent_add_dropout', type=parse_bool, const=True, nargs='?', help='If true, adds a dropout layer to the student model of NoisyStudent.')
    parser.add_argument('--noisystudent_dropout_rate', type=float)
    parser.add_argument('--pretrained_model_path', default=None, type=str, help='Specify a path to pretrained model weights')
    parser.add_argument('--load_featurizer_only', default=False, type=parse_bool, const=True, nargs='?', help='If true, only loads the featurizer weights and not the classifier weights.')

    # NoisyStudent-specific loading
    parser.add_argument('--teacher_model_path', type=str, help='Path to NoisyStudent teacher model weights. If this is defined, pseudolabels will first be computed for unlabeled data before anything else runs.')

    # Transforms
    parser.add_argument('--transform', choices=supported.transforms)
    parser.add_argument('--additional_train_transform', choices=supported.additional_transforms, help='Optional data augmentations to layer on top of the default transforms.')
    parser.add_argument('--target_resolution', nargs='+', type=int, help='The input resolution that images will be resized to before being passed into the model. For example, use --target_resolution 224 224 for a standard ResNet.')
    parser.add_argument('--resize_scale', type=float)
    parser.add_argument('--max_token_length', type=int)
    parser.add_argument('--randaugment_n', type=int, help='Number of RandAugment transformations to apply.')

    # Objective
    parser.add_argument('--loss_function', choices=supported.losses)
    parser.add_argument('--loss_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for loss initialization passed as key1=value1 key2=value2')

    # Algorithm
    parser.add_argument('--groupby_fields', nargs='+')
    parser.add_argument('--group_dro_step_size', type=float)
    parser.add_argument('--coral_penalty_weight', type=float)
    parser.add_argument('--dann_penalty_weight', type=float)
    parser.add_argument('--dann_classifier_lr', type=float)
    parser.add_argument('--dann_featurizer_lr', type=float)
    parser.add_argument('--dann_discriminator_lr', type=float)
    parser.add_argument('--afn_penalty_weight', type=float)
    parser.add_argument('--safn_delta_r', type=float)
    parser.add_argument('--hafn_r', type=float)
    parser.add_argument('--use_hafn', default=False, type=parse_bool, const=True, nargs='?')
    parser.add_argument('--irm_lambda', type=float)
    parser.add_argument('--irm_penalty_anneal_iters', type=int)
    parser.add_argument('--self_training_lambda', type=float)
    parser.add_argument('--self_training_threshold', type=float, default=0.9)
    parser.add_argument('--pseudolabel_T2', type=float, help='Percentage of total iterations at which to end linear scheduling and hold lambda at the max value')
    parser.add_argument('--soft_pseudolabels', default=False, type=parse_bool, const=True, nargs='?')
    parser.add_argument('--algo_log_metric')
    parser.add_argument('--process_pseudolabels_function', choices=supported.process_pseudolabels_functions)

    # PCG
    parser.add_argument('--pcg_ps_update_freq', type=int, help='Gaps between offline pseudo label update in #epochs')
    parser.add_argument('--pcg_pretrain_epochs', type=int, help='When current epoch < this number, use clustering to generate pseudo labels. Othersie use source classifier.')
    parser.add_argument('--pcg_super_contra', type=int)
    parser.add_argument('--pcg_unlabel_bs', type=int)
    parser.add_argument('--pcg_uhead_lr', type=float)
    parser.add_argument('--pc_grad', type=int)
    parser.add_argument('--grad_lambda', type=float)
    parser.add_argument('--consistency_lambda', type=float)
    parser.add_argument('--supcontra_lambda', type=float)
    parser.add_argument('--supcontra_temperature', type=float)
    parser.add_argument('--clustering_epochs', type=int)
    parser.add_argument('--bce_type', choices=['RK', 'cos'], default='RK')
    parser.add_argument('--bce_sim_threshold', type=float)
    parser.add_argument('--bce_diff_threshold', type=float)
    parser.add_argument('--bce_topk', type=int)
    parser.add_argument('--bce_sim_topk', type=int)
    parser.add_argument('--bce_diff_topk', type=int)
    parser.add_argument('--pcg_dropout', type=float)
    parser.add_argument('--use_mlp_head', type=int)
    parser.add_argument('--use_mask', type=int)
    parser.add_argument('--pcg_uhead_path', default='')

    # RegD
    parser.add_argument('--noisy_student', action='store_true', help='use noisy student for consistency loss (default with FixMatch).')
    parser.add_argument('--regd_ps_freq', default=2, type=int, help='Frequency to update pseudo labels when using noisy student.')
    parser.add_argument('--bottleneck_dim', default=256, type=int, help='Dimension of bottleneck layer.')
    parser.add_argument('--regd_dropout', default=0.0, type=float, help='Dropout on feature (for classification head).')
    parser.add_argument('--ts_temperature', default=2.0, type=float, help='Tsallis entropy temperature.')
    parser.add_argument('--ts_alpha', default=1.9, type=float, help='Entropy index of Tsallis entropy.')
    parser.add_argument('--supcon_temperature', default=0.07, type=float, help='Supervised contrastive temperature.')
    parser.add_argument('--w_cluster', default=1.0, type=float, help='weight of cluster loss')
    parser.add_argument('--w_irm', default=0.0, type=float, help='weight of IRM loss')
    parser.add_argument('--w_transfer', default=1.0, type=float, help='weight of transfer loss')
    parser.add_argument('--w_con', default=1.0, type=float, help='weight of consistency loss')
    parser.add_argument('--w_supcon', default=0.0, type=float, help='weight of supervsied contrastive loss')
    parser.add_argument('--consistency_start_epoch', default=0, type=int, help='starting epoch to use IRM loss')
    parser.add_argument('--irm_start_epoch', default=0, type=int, help='starting epoch to use IRM loss')
    parser.add_argument('--back_cluster_start_epoch', default=10, type=int, help='starting epoch to back cluster loss')
    parser.add_argument('--erm_start_epoch', default=0, type=int, help='starting epoch to use erm loss')
    parser.add_argument('--topk', default=5, type=int, help='rank statistics threshold for clustering')
    parser.add_argument('--cosine_threshold', default=0.97, type=float, help='cosine threshold for cluster with cos')
    parser.add_argument('--back_cluster', default=False, action='store_true', help='whether to backward cluster loss')
    parser.add_argument('--bottleneck_cluster', action='store_true', help='use bottleneck feature for clustering')
    parser.add_argument('--supcon_margin', default=0.5, type=float, help='margin to compute supcon loss in regression')

    parser.add_argument('--w_compat', default=0.0, type=float, help='weight of compatibility loss')
    parser.add_argument('--w_compat_u', default=1.0, type=float, help='weight of u2l loss compared to l2u')
    parser.add_argument('--compat_start_epoch', default=2, type=int, help='starting epoch to use compat loss')
    parser.add_argument('--back_compat_l', action='store_true', help='backward l2u compatibility loss to backbone')
    parser.add_argument('--w_compat_irm', default=0.0, type=float, help='weight of compatibility irm loss')
    parser.add_argument('--compat_irm_start_epoch', default=100, type=int, help='starting epoch to use compat loss')
    parser.add_argument('--compat_irm_sim_mode', default='stats', type=str, help='mode to generate target sim matrix for compat irm loss')
    parser.add_argument('--back_compat_irm', action='store_true', help='back compat IRM loss to backbone')

    # regression specific
    parser.add_argument('--compat_sim_margin', default=0.1, type=float, help='regression margin for sim pairs')
    parser.add_argument('--compat_margin', default=0.25, type=float, help='regression margin')

    # classification specific
    parser.add_argument('--cluster_all_compat', action='store_true', help='use cluster all head to compute compat')
    parser.add_argument('--compat_mode', type=str, default='sim', help='gt | stats | sim | fsim')
    parser.add_argument('--compat_sim_threshold', default=-2.0, type=float, help='threshold to deem a pair similar')
    parser.add_argument('--compat_diff_threshold', default=-2.0, type=float, help='threshold to deem a pair different')
    parser.add_argument('--back_compat_u', action='store_true', help='backward u2l compatibility loss to backbone')
    parser.add_argument('--compat_sim_ratio', default=0.1, type=float, help='top percent of pairs similarity treated as sim')
    parser.add_argument('--compat_diff_ratio', default=0.5, type=float, help='bottom percent of pairs similarity treated as diff')
    parser.add_argument('--compat_sim_ratio_delta', default=0.0, type=float, help='increase of sim ratio per epoch')
    parser.add_argument('--compat_diff_ratio_delta', default=0.0, type=float, help='increase of diff ratio per epoch')
    parser.add_argument('--num_pseudo_classes', default=150, type=int, help='for regression/multitask, number of pseudo classes')
    # multitask specific
    parser.add_argument('--compat_multitask_normalize', action='store_true', help='use multitask logits to compute pairwise similarity instead of probs')
    parser.add_argument('--compat_multitask_margin', default=0.0, type=float, help='when multitask, the cosine embed loss margin. typically between 0-0.5')
    # Dim reduction
    parser.add_argument('--dim_reduction', type=str, default='none', help='mode of dimension reduction for feature (used for clustering)')
    parser.add_argument('--reduced_dim', type=int, default=50, help='dim reduction dimension')

    # Model selection
    parser.add_argument('--val_metric')
    parser.add_argument('--val_metric_decreasing', type=parse_bool, const=True, nargs='?')

    # Optimization
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--optimizer', choices=supported.optimizers)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--optimizer_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for optimizer initialization passed as key1=value1 key2=value2')

    # Scheduler
    parser.add_argument('--scheduler', choices=supported.schedulers)
    parser.add_argument('--scheduler_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for scheduler initialization passed as key1=value1 key2=value2')
    parser.add_argument('--scheduler_metric_split', choices=['train', 'val'], default='val')
    parser.add_argument('--scheduler_metric_name')

    # Evaluation
    parser.add_argument('--process_outputs_function', choices = supported.process_outputs_functions)
    parser.add_argument('--evaluate_all_splits', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--eval_splits', nargs='+', default=[])
    parser.add_argument('--eval_only', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--eval_epoch', default=None, type=int, help='If eval_only is set, then eval_epoch allows you to specify evaluating at a particular epoch. By default, it evaluates the best epoch by validation performance.')

    # Misc
    parser.add_argument('--all_seeds', action='store_true', help='Train on all seeds')
    parser.add_argument('--device', type=int, nargs='+', default=[0])
    parser.add_argument('--gpu', type=int, default=-1, help="Used by noisy student wrapper to set GPU")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--auto_log_dir', default=".")
    parser.add_argument('--tensorboard_dir', default='')
    parser.add_argument('--exp_name', default='')
    parser.add_argument('--log_every', default=50, type=int)
    parser.add_argument('--save_step', type=int)
    parser.add_argument('--save_best', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--save_last', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--save_pred', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--no_group_logging', type=parse_bool, const=True, nargs='?')
    parser.add_argument('--progress_bar', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--resume', type=parse_bool, const=True, nargs='?', default=False, help='Whether to resume from the most recent saved model in the current log_dir.')

    # Weights & Biases
    parser.add_argument('--use_wandb', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--wandb_api_key_path', type=str,
                        help="Path to Weights & Biases API Key. If use_wandb is set to True and this argument is not specified, user will be prompted to authenticate.")
    parser.add_argument('--wandb_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for wandb.init() passed as key1=value1 key2=value2')

    config = parser.parse_args()
    config = populate_defaults(config)

    # For the GlobalWheat detection dataset,
    # we need to change the multiprocessing strategy or there will be
    # too many open file descriptors.
    if config.dataset == 'globalwheat':
        torch.multiprocessing.set_sharing_strategy('file_system')

    # Set device
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if len(config.device) > device_count:
            raise ValueError(f"Specified {len(config.device)} devices, but only {device_count} devices found.")

        config.use_data_parallel = len(config.device) > 1
        device_str = ",".join(map(str, config.device))
        os.environ["CUDA_VISIBLE_DEVICES"] = device_str
        config.device = torch.device("cuda")
    else:
        config.use_data_parallel = False
        config.device = torch.device("cpu")
    
    if not config.all_seeds:
        logger = Logger(None)
        train_info = main(config, logger)

        log_file = config.auto_log_dir
        if config.dataset == 'globalwheat' and config.algorithm == 'RegDet':
            with open(log_file, "a") as file:
                if config.seed == 0:
                    for name, val in vars(config).items():
                        file.write(f'{name.replace("_"," ").capitalize()}: {val}, ')
                    file.write('\n')
                file.write(str(train_info))
                file.write('\n')
                if config.seed == 2:
                    file.write('\n\n')
    else:
        # Train on all seeds
        assert not config.eval_only
        if config.dataset == 'camelyon17':
            seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        elif config.dataset == 'iwildcam' or config.dataset == 'fmow':
            seeds = [0, 1, 2]
        elif config.dataset == 'poverty':
            seeds = ['A', 'B', 'C', 'D', 'E']
        elif config.dataset == 'civilcomments':
            seeds = [0, 1, 2, 3, 4]
        elif config.dataset == 'amazon':
            seeds = [0, 1, 2]
        elif config.dataset == 'ogb-molpcba':
            seeds = [0, 1, 2]
        elif config.dataset == 'globalwheat':
            seeds = [0, 1, 2]
        else:
            assert False    # Not implemented yet
        train_info_list = []

        logger = Logger(None)
        i = -1
        original_log_dir = config.log_dir
        for seed in seeds:
            i += 1
            if config.dataset != 'poverty':
                config.seed = seed
            else:
                config.dataset_kwargs['fold'] = seed
                if config.pretrained_model_path is not None and i != 0:
                    config.pretrained_model_path = config.pretrained_model_path.replace('fold:%s' % seeds[i-1], 'fold:%s' % seeds[i])
                if config.teacher_model_path is not None and i != 0:
                    config.teacher_model_path = config.teacher_model_path.replace('fold:%s' % seeds[i-1], 'fold:%s' % seeds[i])

            if 'RegD' in config.algorithm:
                config.log_dir = original_log_dir   # RegD updates log dir
            else:
                config.log_dir = original_log_dir + f"_s{seed}"

            train_info = main(config, logger)
            train_info_dir = os.path.join(config.log_dir, "train_info.pkl")
            with open(train_info_dir, 'wb') as f:
                pickle.dump(train_info, f)
            train_info_list.append(train_info)
        
        # Process train info
        train_info_dir = os.path.join(config.log_dir, "final_train_info_list.pkl")
        with open(train_info_dir, 'wb') as f:
            pickle.dump(train_info_list, f)
        
        print_train_info(train_info_list, config)
        logger.close()