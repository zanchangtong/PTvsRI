import sys
import argparse
import os
import math
import yaml
import copy
import re
import norm
import sacrebleu
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from fairseq import utils
from fairseq import checkpoint_utils, options, progress_bar, utils

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

def get_loss(model, criterion, batch_iterator):
    batch_num = 512
    loss = 0
    sample_sum = 0
    model.eval()
    criterion.eval()
    iter = batch_iterator.next_epoch_itr(shuffle=False)
    
    with torch.no_grad():
        for idx, sample in enumerate(iter):
            if idx >= batch_num:
                break
            sample = utils.move_to_cuda(sample)
            loss_ , sample_size, _ = criterion(model, sample)
            loss = loss + loss_
            sample_sum = sample_sum + sample_size
            del loss_
    return (loss/sample_sum/math.log(2)).data 

def normalize_filter(bs, ws):
    bs = {k: v.float() for k, v in bs.items()}
    ws = {k: v.float() for k, v in ws.items()}

    norm_bs = {}
    for k in bs:
        ws_norm = torch.norm(ws[k], dim=0, keepdim=True)
        bs_norm = torch.norm(bs[k], dim=0, keepdim=True)
        norm_bs[k] = ws_norm / (bs_norm + 1e-7) * bs[k]

    return norm_bs

def ignore_bn(ws):
    ignored_ws = {}
    for k in ws:
        if len(ws[k].size()) < 2:
            ignored_ws[k] = torch.zeros(size=ws[k].size(), device=ws[k].device)
        else:
            ignored_ws[k] = ws[k]
    return ignored_ws


def ignore_running_stats(ws):
    return ignore_kw(ws, ["num_batches_tracked"])


def ignore_kw(ws, kws=None):
    kws = [] if kws is None else kws

    ignored_ws = {}
    for k in ws:
        if any([re.search(kw, k) for kw in kws]):
            ignored_ws[k] = torch.zeros(size=ws[k].size(), device=ws[k].device)
        else:
            ignored_ws[k] = ws[k]
    return ignored_ws

def rand_basis(ws, gpu=True):
    return {k: torch.randn(size=v.shape, device="cuda" if gpu else None) for k, v in ws.items()}

def create_bases(model, kws=None, gpu=True):
    kws = [] if kws is None else kws
    ws0 = copy.deepcopy(model.state_dict())
    bases = [rand_basis(ws0, gpu) for _ in range(2)]  
    bases = [normalize_filter(bs, ws0) for bs in bases]
    bases = [ignore_bn(bs) for bs in bases]
    bases = [ignore_kw(bs, kws) for bs in bases]

    return bases

def get_loss_landscape(test,func,
                       bases=None, kws=None,
                       cutoffs=(0.0, 0.9), bins=np.linspace(0.0, 1.0, 11), verbose=False, period=10, gpu=True,
                       x_min=-1.0, x_max=1.0, n_x=11, y_min=-1.0, y_max=1.0, n_y=11):
    test.model = test.model.cuda() if gpu else test.model.cpu()
    test.model = copy.deepcopy(test.model)
    ws0 = copy.deepcopy(test.model.state_dict())
    kws = [] if kws is None else kws
    bases = create_bases(test.model, kws, gpu) if bases is None else bases
    xs = np.linspace(x_min, x_max, n_x)
    ys = np.linspace(y_min, y_max, n_y)
    ratio_grid = np.stack(np.meshgrid(xs, ys), axis=0).transpose((1, 2, 0))

    metrics_grid = {}
    for ratio in ratio_grid.reshape([-1, 2]):
        ws = copy.deepcopy(ws0)
        gs = [{k: r * bs[k] for k in bs} for r, bs in zip(ratio, bases)]
        gs = {k: torch.sum(torch.stack([g[k] for g in gs]), dim=0) + ws[k] for k in gs[0]}
        test.model.load_state_dict(gs)
        del gs
        print("Grid: ", ratio, end=", ")
        loss = func()
        print(loss)
        metrics_grid[tuple(ratio)] = loss

    return metrics_grid

def aux_basis(ws1, ws0, gpu=True):
    basis={}
    for k, v in ws0.items():
        if 'embed_tokens' in k or 'embed_positions' in k or 'output_projection' in k:
            basis[k] = v
            continue
        else:
            basis[k] = ws1[k]-v
    return basis

def create_bases_v2(tgt_model, kws=None, gpu=True, init_model=None, aux_model=None):
    kws = [] if kws is None else kws
    ws0 = copy.deepcopy(init_model.state_dict())
    ws1 = copy.deepcopy(tgt_model.state_dict())
    ws2 = copy.deepcopy(aux_model.state_dict())
    if gpu:
        ws0 = {k: v.cuda() for k, v in ws0.items()}
        ws2 = {k: v.cuda() for k, v in ws2.items()}
        
    bases = [aux_basis(ws1, ws0, gpu), aux_basis(ws2, ws0, gpu)]  # Use two bases
    bases[1] = normalize_filter(bases[1], bases[0])
    bases = [ignore_bn(bs) for bs in bases]
    bases = [ignore_kw(bs, kws) for bs in bases]
    return bases

def get_loss_landscape_with_aux_model(test_model,func,initial_model,auxiliary_model,criterion, batch_iterator,
                       bases=None, kws=None,
                       cutoffs=(0.0, 0.9), bins=np.linspace(0.0, 1.0, 11), verbose=False, period=10, gpu=True,
                       x_min=-1.0, x_max=1.0, n_x=11, y_min=-1.0, y_max=1.0, n_y=11):
    test_model = test_model.cuda() if gpu else test_model.cpu()
    test_model = copy.deepcopy(test_model)
    ws0 = copy.deepcopy(test_model.state_dict())
    ws_init = copy.deepcopy(initial_model.state_dict())
    kws = [] if kws is None else kws
    bases = create_bases_v2(test_model, kws, gpu, initial_model, auxiliary_model) if bases is None else bases
    xs = np.linspace(x_min, x_max, n_x)
    ys = np.linspace(y_min, y_max, n_y)
    ratio_grid = np.stack(np.meshgrid(xs, ys), axis=0).transpose((1, 2, 0))

    metrics_grid = {}
    for ratio in ratio_grid.reshape([-1, 2]):
        ratio_shift = [ ratio[0] - 1, ratio[1]]
        ws = copy.deepcopy(ws0)
        gs = [{k: r * bs[k] for k in bs} for r, bs in zip(ratio_shift, bases)]
        gs = {k: torch.sum(torch.stack([g[k] for g in gs]), dim=0) + ws[k] for k in gs[0]}
        test_model.load_state_dict(gs)
        del gs
        print("Grid: ", ratio, end=", ")
        loss = func(test_model, criterion, batch_iterator)
        print(loss)
        metrics_grid[tuple(ratio)] = loss
    return metrics_grid


if __name__=='__main__':
    output_dir="/loss_landscape"
    init_model_path ='trimed_mbart.pt'
    final_model_path ='ende_RI_trimed_dict.pt'
    aux_model_path="ende_PT_trimed_dict.pt"

    scale = 3.0
    n = 61
    
    overrides = None
    test_model, model_args, task = checkpoint_utils.load_model_ensemble_and_task(
        [final_model_path],
        arg_overrides=overrides,
        )
    test_model = test_model[0]
    init_model, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [init_model_path],
        arg_overrides=overrides,
        )
    init_model = init_model[0]
    aux_model, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [aux_model_path],
        arg_overrides=overrides,
        )
    aux_model = aux_model[0]

    criterion = task.build_criterion(model_args.criterion)
    subset = 'valid'
    task.load_dataset(subset, combine=False, epoch=0)
    dataset = task.dataset(subset)

    batch_itr = task.get_batch_iterator(
                dataset=dataset,
                max_tokens=2048,
                max_positions=utils.resolve_max_positions(
                        task.max_positions(),
                        *[m.max_positions() for m in [test_model]],
                        ),
                        ignore_invalid_inputs=True,
                        )

    metrics_grid = get_loss_landscape_with_aux_model(
        test_model, get_loss, initial_model=init_model, auxiliary_model=aux_model,
        criterion=criterion, batch_iterator=batch_itr,
        kws=["embed_tokens", "embed_positions", 'output_projection'],
        x_min=-1.0 * scale, x_max=1.0 * scale, n_x=n, y_min=-1.0 * scale, y_max=1.0 * scale, n_y=n, gpu=True,
    )
    np.save(output_dir + '/wmt19ende.RI.lls.npy', metrics_grid)


    # draw lls
    import math
    import matplotlib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import cm

    def draw(datas):
        def prepare_data(data, scale_p = 1):
            p = int(math.sqrt(len(data["x"])))
            shape = [p, p]
            xs = np.array(data["x"]).reshape(shape) 
            ys = np.array(data["y"]).reshape(shape)
            for i, point in enumerate(data['loss']):
                data['loss'][i] = data['loss'][i] * scale_p
            zs = np.array(data["loss"]).reshape(shape)
            
            def cut(items):
                for idx, item in enumerate(items):
                    if idx == 0:
                        items_ = [item[10:]]
                    else:
                        items_.append(item[10:])

                return np.array(items_)
            
            return xs, ys, zs

        for idx, data in enumerate(datas):

            datas[idx]["x"], datas[idx]["y"], datas[idx]["loss"] = prepare_data(data)
        

        xs, ys, zs = datas[0]["x"], datas[0]["y"], datas[0]["loss"]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=55, azim=40)  # angle

        norm = matplotlib.colors.Normalize(vmin=0, vmax=65)
        levels = np.arange(0, 70, 10)
        surf = ax.plot_surface(
            xs, ys, zs, 
            rstride = 1, cstride = 1, cmap = plt.cm.coolwarm, norm = norm, linewidth=0, antialiased=False) # 网格
        ax.contour( xs, ys, zs, zdir='x', offset=-3, cmap="rainbow")
        plt.rcParams['font.size'] = 27
        fig.colorbar(surf, shrink=0.49, aspect=15)
        
        adjust_lim = 3.5 # 0.8
        ax.set_xlim(-1 * 2.5, 1 * 3.5)
        ax.set_ylim(-1 * 3.5, 1 * 3.5)
        ax.set_zlim(0, 75)
        plt.tick_params(labelsize=27)
        
        plt.savefig("Fusion_lls.png")
        plt.show()

    datas = np.load(output_dir + '/wmt19ende.RI.lls.npy', allow_pickle=True).item()
    datas_ = [{"x":[], "y":[], "loss":[]}]
    for key in datas.keys():
        datas_[0]["x"].append(key[0])
        datas_[0]["y"].append(key[1])
        datas_[0]["loss"].append(datas[key].cpu())
    
    draw(datas_)

