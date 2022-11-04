import copy
import os
import torch
from tqdm import tqdm
import math
from torch.optim import SGD, Adam
from configs.supported import process_outputs_functions, process_pseudolabels_functions
from utils import save_model, save_pred, get_pred_prefix, get_model_prefix, collate_list, detach_and_clone, InfiniteDataIterator
from utils import BCE, PairEnum, AverageMeter
from torch.optim.lr_scheduler import ExponentialLR
from utils import Visualizer
import torch.nn.functional as F

def run_epoch(algorithm, dataset, general_logger, epoch, config, train, unlabeled_dataset=None):
    if dataset['verbose']:
        general_logger.write(f"\n{dataset['name']}:\n")

    # Not preallocating memory is slower
    # but makes it easier to handle different types of data loaders
    # (which might not return exactly the same number of examples per epoch)
    epoch_y_true = []
    epoch_y_pred = []
    epoch_metadata = []

    # Assert that data loaders are defined for the datasets
    assert 'loader' in dataset, "A data loader must be defined for the dataset."
    if unlabeled_dataset:
        assert 'loader' in unlabeled_dataset, "A data loader must be defined for the dataset."

    if train:
        algorithm.train()
        torch.set_grad_enabled(True)
    else:
        algorithm.eval()
        torch.set_grad_enabled(False)
    
    if train and hasattr(algorithm, 'start_of_epoch_setup'):
        algorithm.start_of_epoch_setup(epoch)

    if train:
        algorithm.train()
        torch.set_grad_enabled(True)
    else:
        algorithm.eval()
        torch.set_grad_enabled(False)

    batches = dataset['loader']
    if config.progress_bar:
        batches = tqdm(batches)
    last_batch_idx = len(batches)-1

    if unlabeled_dataset:
        unlabeled_data_iterator = InfiniteDataIterator(unlabeled_dataset['loader'])

    # Using enumerate(iterator) can sometimes leak memory in some environments (!)
    # so we manually increment batch_idx
    batch_idx = 0
    for labeled_batch in batches:
        if train:
            if unlabeled_dataset:
                unlabeled_batch = next(unlabeled_data_iterator)
                batch_results = algorithm.update(labeled_batch, unlabeled_batch, is_epoch_end=(batch_idx==last_batch_idx))
            else:
                batch_results = algorithm.update(labeled_batch, is_epoch_end=(batch_idx==last_batch_idx))
        else:
            batch_results = algorithm.evaluate(labeled_batch)

        # These tensors are already detached, but we need to clone them again
        # Otherwise they don't get garbage collected properly in some versions
        # The extra detach is just for safety
        # (they should already be detached in batch_results)
        epoch_y_true.append(detach_and_clone(batch_results['y_true']))
        y_pred = detach_and_clone(batch_results['y_pred'])
        if config.process_outputs_function is not None:
            y_pred = process_outputs_functions[config.process_outputs_function](y_pred)
        epoch_y_pred.append(y_pred)
        epoch_metadata.append(detach_and_clone(batch_results['metadata']))

        if train: 
            effective_batch_idx = (batch_idx + 1) / config.gradient_accumulation_steps
        else: 
            effective_batch_idx = batch_idx + 1

        if train and effective_batch_idx % config.log_every==0:
            log_results(algorithm, dataset, general_logger, epoch, math.ceil(effective_batch_idx))

        batch_idx += 1

    epoch_y_pred = collate_list(epoch_y_pred)
    epoch_y_true = collate_list(epoch_y_true)
    epoch_metadata = collate_list(epoch_metadata)

    results, results_str = dataset['dataset'].eval(
        epoch_y_pred,
        epoch_y_true,
        epoch_metadata)

    if config.scheduler_metric_split==dataset['split']:
        algorithm.step_schedulers(
            is_epoch=True,
            metrics=results,
            log_access=(not train))

    # log after updating the scheduler in case it needs to access the internal logs
    log_results(algorithm, dataset, general_logger, epoch, math.ceil(effective_batch_idx))

    results['epoch'] = epoch
    dataset['eval_logger'].log(results)
    if dataset['verbose']:
        general_logger.write('Epoch eval:\n')
        general_logger.write(results_str)

    return results, epoch_y_pred


def train(algorithm, datasets, general_logger, config, epoch_offset, best_val_metric, unlabeled_dataset=None):
    """
    Train loop that, each epoch:
        - Steps an algorithm on the datasets['train'] split and the unlabeled split
        - Evaluates the algorithm on the datasets['val'] split
        - Saves models / preds with frequency according to the configs
        - Evaluates on any other specified splits in the configs
    Assumes that the datasets dict contains labeled data.
    """
    algorithm.prefix = get_model_prefix(datasets['val'], config)
    algorithm.logger = general_logger
    test_best = 0.0
    test_metrics = []
    val_metrics = []
    for epoch in range(epoch_offset, config.n_epochs):
        general_logger.write('\nEpoch [%d]:\n' % epoch)

        # First run training
        algorithm.current_epoch = epoch
        run_epoch(algorithm, datasets['train'], general_logger, epoch, config, train=True, unlabeled_dataset=unlabeled_dataset)

        # Then run val
        val_results, y_pred = run_epoch(algorithm, datasets['val'], general_logger, epoch, config, train=False)
        curr_val_metric = val_results[config.val_metric]
        general_logger.write(f'Validation {config.val_metric}: {curr_val_metric:.3f}\n')

        if hasattr(algorithm, "visualizer"):
            algorithm.visualizer.plot_items(epoch, {"z_val_acc": curr_val_metric})
        val_metrics.append(curr_val_metric)

        if best_val_metric is None:
            is_best = True
        else:
            if config.val_metric_decreasing:
                is_best = curr_val_metric < best_val_metric
            else:
                is_best = curr_val_metric > best_val_metric
        if is_best:
            best_val_metric = curr_val_metric
            general_logger.write(f'Epoch {epoch} has the best validation performance so far.\n')

        save_model_if_needed(algorithm, datasets['val'], epoch, config, is_best, best_val_metric)
        save_pred_if_needed(y_pred, datasets['val'], epoch, config, is_best)

        # Then run everything else
        if config.evaluate_all_splits:
            additional_splits = [split for split in datasets.keys() if split not in ['train','val']]
        else:
            additional_splits = config.eval_splits
        for split in additional_splits:
            results, y_pred = run_epoch(algorithm, datasets[split], general_logger, epoch, config, train=False)
            save_pred_if_needed(y_pred, datasets[split], epoch, config, is_best)
            if split == 'test':
                if is_best:
                    test_best = results[config.val_metric]
                test_metrics.append(results[config.val_metric])
            if hasattr(algorithm, "visualizer"):
                curr_val_metric = results[config.val_metric]
                algorithm.visualizer.plot_items(epoch, {"z_%s acc" % split: curr_val_metric})
        if hasattr(algorithm, "visualizer"):
            algorithm.visualizer.plot_items(epoch, {"z_test_best": test_best})
        general_logger.write('\n')
        
    train_info = {
        "test": test_metrics,
        "val": val_metrics
    }
    return train_info


def evaluate(algorithm, datasets, epoch, general_logger, config, is_best):
    algorithm.eval()
    torch.set_grad_enabled(False)
    for split, dataset in datasets.items():
        if (not config.evaluate_all_splits) and (split not in config.eval_splits):
            continue
        epoch_y_true = []
        epoch_y_pred = []
        epoch_metadata = []
        iterator = tqdm(dataset['loader']) if config.progress_bar else dataset['loader']
        for batch in iterator:
            batch_results = algorithm.evaluate(batch)
            epoch_y_true.append(detach_and_clone(batch_results['y_true']))
            y_pred = detach_and_clone(batch_results['y_pred'])
            if config.process_outputs_function is not None:
                y_pred = process_outputs_functions[config.process_outputs_function](y_pred)
            epoch_y_pred.append(y_pred)
            epoch_metadata.append(detach_and_clone(batch_results['metadata']))

        epoch_y_pred = collate_list(epoch_y_pred)
        epoch_y_true = collate_list(epoch_y_true)
        epoch_metadata = collate_list(epoch_metadata)
        results, results_str = dataset['dataset'].eval(
            epoch_y_pred,
            epoch_y_true,
            epoch_metadata)

        results['epoch'] = epoch
        dataset['eval_logger'].log(results)
        general_logger.write(f'Eval split {split} at epoch {epoch}:\n')
        general_logger.write(results_str)

        # Skip saving train preds, since the train loader generally shuffles the data
        if split != 'train':
            save_pred_if_needed(epoch_y_pred, dataset, epoch, config, is_best, force_save=True)

def infer_predictions(model, loader, config, two_views=False):
    """
    Simple inference loop that performs inference using a model (not algorithm) and returns model outputs.
    Compatible with both labeled and unlabeled WILDS datasets.
    """
    print("Infering pseudo labels for unlabel data.")
    model.eval()
    y_pred = []
    iterator = tqdm(loader) if config.progress_bar else loader
    for batch in iterator:
        if not two_views:
            x = batch[0]
        else:
            (x_weak, x_strong) = batch[0]
            x = x_weak
        x = x.to(config.device)
        with torch.no_grad(): 
            output = model(x)
            if not config.soft_pseudolabels and config.process_pseudolabels_function is not None:
                _, output, _, _ = process_pseudolabels_functions[config.process_pseudolabels_function](
                    output,
                    confidence_threshold=config.self_training_threshold if config.dataset == 'globalwheat' else 0
                )
            elif config.soft_pseudolabels:
                output = torch.nn.functional.softmax(output, dim=1)
        if isinstance(output, list):
            y_pred.extend(detach_and_clone(output))
        else:
            y_pred.append(detach_and_clone(output))

    return torch.cat(y_pred, 0) if torch.is_tensor(y_pred[0]) else y_pred

def learn_pcg_unlabel_head(featurizer, featurizer_d_out, d_out, loader, config, u_head=None, cluster_steps=0):
    if config.clustering_epochs > 0:
        print ('Start clustering for unlabel pseudo-labels for %d epochs' % config.clustering_epochs)
    if u_head is None:
        classifier = torch.nn.Linear(featurizer_d_out, d_out).to(config.device)
    else:
        classifier = u_head.to(config.device)
    optimizer = SGD(classifier.parameters(), lr=config.pcg_uhead_lr, weight_decay=config.weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    criterion_bce = BCE()
    step = cluster_steps
    visualizer = Visualizer(config.tensorboard_dir, config.exp_name)
    for epoch in range(config.clustering_epochs):
        loss_record = AverageMeter()
        featurizer.eval()
        classifier.train()
        # w = config.rampup_coefficient * sigmoid_rampup(epoch, config.rampup_length)

        iterator = tqdm(loader) if config.progress_bar else loader
        for (x_weak, x_strong), metadata in iterator:
            step += 1
            x_weak, x_strong = x_weak.to(config.device), x_strong.to(config.device)
            with torch.no_grad():
                feat = featurizer(x_weak)
            output = classifier(feat.detach())
            with torch.no_grad():
                feat_bar = featurizer(x_strong)
            output_bar = classifier(feat_bar.detach())

            prob, prob_bar = F.softmax(output, dim=1), F.softmax(output_bar, dim=1)

            rank_feat = feat
            if config.bce_type == 'cos':
                # default: cosine similarity with threshold
                feat_row, feat_col = PairEnum(F.normalize(rank_feat, dim=1))
                tmp_distance_ori = torch.bmm(feat_row.view(feat_row.size(0), 1, -1), feat_col.view(feat_row.size(0), -1, 1))
                tmp_distance_ori = tmp_distance_ori.squeeze()
                target_ulb = torch.zeros_like(tmp_distance_ori).float() - 1
                target_ulb[tmp_distance_ori > config.bce_cosine_threshold] = 1
            elif config.bce_type == 'RK':
                # top-k rank statics
                rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
                rank_idx1, rank_idx2 = PairEnum(rank_idx)
                rank_idx1, rank_idx2 = rank_idx1[:, :config.bce_topk], rank_idx2[:, :config.bce_topk]
                rank_idx1, _ = torch.sort(rank_idx1, dim=1)
                rank_idx2, _ = torch.sort(rank_idx2, dim=1)
                rank_diff = rank_idx1 - rank_idx2
                rank_diff = torch.sum(torch.abs(rank_diff), dim=1)
                target_ulb = torch.ones_like(rank_diff).float().to(config.device)
                target_ulb[rank_diff > 0] = -1

            prob1_ulb, _ = PairEnum(prob)
            _, prob2_ulb = PairEnum(prob_bar)

            loss_bce = criterion_bce(prob1_ulb, prob2_ulb, target_ulb)
            loss_record.update(loss_bce.item(), x_weak.size(0))
            visualize_items = {
                "BCE loss": loss_bce
            }
            visualizer.plot_items(step, visualize_items)
            # ===================backward=====================
            optimizer.zero_grad()
            loss_bce.backward()
            optimizer.step()
        scheduler.step()
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
    
    torch.save(classifier.state_dict(), os.path.join(config.log_dir, "pcg_unlabel_head.pth"))
    return classifier, step

def log_results(algorithm, dataset, general_logger, epoch, effective_batch_idx):
    if algorithm.has_log:
        log = algorithm.get_log()
        log['epoch'] = epoch
        log['batch'] = effective_batch_idx
        dataset['algo_logger'].log(log)
        if dataset['verbose']:
            general_logger.write(algorithm.get_pretty_log_str())
        algorithm.reset_log()


def save_pred_if_needed(y_pred, dataset, epoch, config, is_best, force_save=False):
    if config.save_pred:
        prefix = get_pred_prefix(dataset, config)
        if force_save or (config.save_step is not None and (epoch + 1) % config.save_step == 0):
            save_pred(y_pred, prefix + f'epoch:{epoch}_pred')
        if (not force_save) and config.save_last:
            save_pred(y_pred, prefix + f'epoch:last_pred')
        if config.save_best and is_best:
            save_pred(y_pred, prefix + f'epoch:best_pred')


def save_model_if_needed(algorithm, dataset, epoch, config, is_best, best_val_metric):
    prefix = get_model_prefix(dataset, config)
    if config.save_step is not None and (epoch + 1) % config.save_step == 0:
        save_model(algorithm, epoch, best_val_metric, prefix + f'epoch:{epoch}_model.pth')
    if config.save_last:
        save_model(algorithm, epoch, best_val_metric, prefix + 'epoch:last_model.pth')
    if config.save_best and is_best:
        save_model(algorithm, epoch, best_val_metric, prefix + 'epoch:best_model.pth')
