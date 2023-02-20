import sys
import torch
import numpy as np
import transformers
import routines
from torch import nn
import copy
from fairseq.models.bart import BARTModel
from fairseq.models.transformer import TransformerModel
import fairseq
fairseq.modules.MultiheadAttention.functional_self_attention = False

def get_args(**new_args):
    default_args = {
        "model_name": "mlpnet",
        "gpu_id": -1,
        "disable_bias": True,
        # "dist_normalize": False,
        "geom_ensemble_type": "acts",
        "normalize_acts": "no", # get_activation
        "act_num_samples": 200,
        "activation_seed": 0,
        "activation_mode": "raw",
        "debug": False,
        "width_ratio": 1,
        "prelu_acts": True,
        "pool_acts": False, "pool_relu": False,
    }
    data_args = {
        "dataset": "mnist",
        "batch_size_train": 256,
        "batch_size_test": 256,
        "to_download": True,
        "skip_idxs": [],
        "personal_class_idx": 9,
        "seed": 1,
    }
    importance_args = {
        "importance": "uniform",
        "importance_method": "uniform",  # none
        "softmax_temperature": 10 # importance==acts
    }
    ground_metric_args = {
        "clip_gm": False,
        "ground_metric_normalize": "none",
        "ground_metric": "euclidean",
        "normalize_coords": False,  # mectric
        "dtype": torch.float64,
        "mem_efficient": False
    }
    ot_args = {
        "gromov": False,
        "reg": 0.001,
        "proper_marginals": True,
        "exact": True,
    }
    retrain_args = {
        "rt_lr": 0.01,
        "rt_lr_decay": 1,
        "rt_seed": 1,
        "rt_momentum": 0.9,
        "rt_epoch": 40,
        "retrain_lr_decay_factor": 2,
        "retrain_lr_decay_epochs": "15_15_10",
        "rt_log_interval": 100,
        "rt_result_dir" : '/workspace/zhangzheng/otfusion/checkpoints',
        "rt_exp_name": 'exp_name',
        'rt_model_save': False,
        "rt_batch_size": 128
    }
    args_dict={   
        **default_args,
        **data_args,
        **importance_args,
        **ground_metric_args,
        **ot_args,
        **retrain_args,
        # "num_hidden_nodes1": 400,
        # "num_hidden_nodes2": 200,
        # "num_hidden_nodes3": 100,
        **new_args
    }

    class X:
        def __init__(self, args_dict) -> None:
            for k,v in args_dict.items():
                self.__dict__[k]= v
    return X(args_dict)

def mbart_translation_test(exp='enzh', aux_task='bpe_from_scratch_4', tgt_task='128X'):

    if exp == 'enzh':
        tgt_mbart = BARTModel.from_pretrained(
            '/workspace/data/users/zanchangtong1/2_High_Resource_Translation/checkpoint_backup/{}'.format(tgt_task),
            checkpoint_file='checkpoint_best.pt',
            )
        aux_mbart = BARTModel.from_pretrained(
            '/workspace/data/users/zanchangtong1/2_High_Resource_Translation/checkpoint_backup/{}'.format(aux_task),
            checkpoint_file='checkpoint_best.pt',
        ) 
        for (task, model) in [(aux_task, aux_mbart), (tgt_task, tgt_mbart)]:
            model.task.args.data = '/workspace/data/users/zanchangtong1/data-in/en_XX-zh_CN_small'
        
    elif exp == 'ende':
        tgt_mbart = BARTModel.from_pretrained(
            '/workspace/data/users/zanchangtong1/2_High_Resource_Translation/checkpoint_backup/{}'.format(tgt_task),
            checkpoint_file='checkpoint_best.pt',
        )
        aux_mbart = BARTModel.from_pretrained(
            '/workspace/data/users/zanchangtong1/2_High_Resource_Translation/checkpoint_backup/{}'.format(aux_task),
            checkpoint_file='checkpoint_avg5.pt',
        ) 
        for (task, model) in [(aux_task, aux_mbart), (tgt_task, tgt_mbart)]:
            model.task.args.data = '/workspace/data/users/zanchangtong1/data-in/en_XX-de_DE_small'
        pass
    
    seed = 222
    max_tokens = 16384
    shard_batch_itr = False
    epoch = 1
    disable_iterator_cache=False
    from fairseq import utils
    tgt_mbart.task.load_dataset('valid', combine=False, epoch=1)
    batch_iterator_tgt = tgt_mbart.task.get_batch_iterator(
            dataset=tgt_mbart.task.dataset(tgt_mbart.cfg.dataset.valid_subset),
            max_tokens=max_tokens,
            max_sentences=tgt_mbart.cfg.dataset.batch_size,
            max_positions=utils.resolve_max_positions(
                tgt_mbart.task.max_positions(),
                tgt_mbart.model.max_positions(),
                max_tokens,
            ),
            ignore_invalid_inputs=True,
            required_batch_size_multiple=tgt_mbart.cfg.dataset.required_batch_size_multiple,
            seed=seed,
            num_shards=tgt_mbart.data_parallel_world_size if shard_batch_itr else 1,
            shard_id=tgt_mbart.data_parallel_rank if shard_batch_itr else 0,
            num_workers=0,
            epoch=epoch,
            data_buffer_size=tgt_mbart.cfg.dataset.data_buffer_size,
            disable_iterator_cache=disable_iterator_cache,
        )

    aux_mbart.task.load_dataset('valid', combine=False, epoch=1)
    batch_iterator_aux = aux_mbart.task.get_batch_iterator(
        dataset=aux_mbart.task.dataset(aux_mbart.cfg.dataset.valid_subset),
        max_tokens=max_tokens,
        max_sentences=aux_mbart.cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            aux_mbart.task.max_positions(),
            aux_mbart.model.max_positions(),
            max_tokens,
        ),
        ignore_invalid_inputs=True,
        required_batch_size_multiple=aux_mbart.cfg.dataset.required_batch_size_multiple,
        seed=seed,
        num_shards=aux_mbart.data_parallel_world_size if shard_batch_itr else 1,
        shard_id=aux_mbart.data_parallel_rank if shard_batch_itr else 0,
        num_workers=0,
        epoch=epoch,
        data_buffer_size=aux_mbart.cfg.dataset.data_buffer_size,
        disable_iterator_cache=disable_iterator_cache,
    )

    model_tgt = tgt_mbart.model
    model_aux = aux_mbart.model
    tgt_inputs = next(batch_iterator_tgt.next_epoch_itr())['net_input']
    aux_inputs = next(batch_iterator_aux.next_epoch_itr())['net_input']
    mlp_args = {
        'model_name': 'mbart',
    }
    args = get_args(**mlp_args)
    return args, model_aux, model_tgt, aux_inputs, tgt_inputs


if __name__ == '__main__':

    translation_test()
