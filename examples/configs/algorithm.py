algorithm_defaults = {
    'ERM': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
        'randaugment_n': 2,     # When running ERM + data augmentation
    },
    'groupDRO': {
        'train_loader': 'standard',
        'uniform_over_groups': True,
        'distinct_groups': True,
        'eval_loader': 'standard',
        'group_dro_step_size': 0.01,
    },
    'deepCORAL': {
        'train_loader': 'group',
        'uniform_over_groups': True,
        'distinct_groups': True,
        'eval_loader': 'standard',
        'coral_penalty_weight': 1.,
        'randaugment_n': 2,
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled & unlabeled examples
    },
    'IRM': {
        'train_loader': 'group',
        'uniform_over_groups': True,
        'distinct_groups': True,
        'eval_loader': 'standard',
        'irm_lambda': 100.,
        'irm_penalty_anneal_iters': 500,
    },
    'DANN': {
        'train_loader': 'group',
        'uniform_over_groups': True,
        'distinct_groups': True,
        'eval_loader': 'standard',
        'randaugment_n': 2,
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled & unlabeled examples
    },
    'AFN': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
        'use_hafn': False,
        'afn_penalty_weight': 0.01,
        'safn_delta_r': 1.0,
        'hafn_r': 1.0,
        'additional_train_transform': 'randaugment',    # Apply strong augmentation to labeled & unlabeled examples
        'randaugment_n': 2,
    },
    'FixMatch': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
        'self_training_lambda': 1,
        'self_training_threshold': 0.7,
        'scheduler': 'FixMatchLR',
        'randaugment_n': 2,
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled examples
    },
    'PseudoLabel': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
        'self_training_lambda': 1,
        'self_training_threshold': 0.7,
        'pseudolabel_T2': 0.4,
        'scheduler': 'FixMatchLR',
        'randaugment_n': 2,
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled & unlabeled examples
    },
    'NoisyStudent': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
        'noisystudent_add_dropout': True,
        'noisystudent_dropout_rate': 0.5,
        'scheduler': 'FixMatchLR',
        'randaugment_n': 2,
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled & unlabeled examples
    },
    'RegD': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
        'scheduler': 'FixMatchLR',
        'randaugment_n': 2,
        'additional_train_transform': 'regd',     # (weak aug, strong aug)
    },
    'RegDNA': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
        'scheduler': 'FixMatchLR',
        'self_training_lambda': 1,
        'self_training_threshold': 0.7,
        'pseudolabel_T2': 0.4,
        'randaugment_n': 2,
        'additional_train_transform': 'randaugment',
    },
    'RegDet': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
        'noisystudent_add_dropout': True,
        'noisystudent_dropout_rate': 0.5,
        'scheduler': 'FixMatchLR',
        'randaugment_n': 2,
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled & unlabeled examples
    },
    'PCG': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
        'randaugment_n': 2,
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled & unlabeled examples
        'pcg_super_contra': 0,
        'pc_grad': 1,
        'grad_lambda': 1.0,
        'consistency_lambda': 1.0,
        'supcontra_lambda': 1.0,
        'bce_type': 'RK',
        'bce_cosine_threshold': 0.95,
        'bce_topk': 15,
        'pcg_uhead_lr': 0.01,
        'pcg_dropout': 0.5,
        'use_mlp_head': 1,
        'pcg_ps_update_freq': 2,
        'pcg_pretrain_epochs': 0,
        'use_mask': False,
        'supcontra_temperature': 0.1
    }
}
