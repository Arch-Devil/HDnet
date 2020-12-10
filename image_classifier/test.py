# -*- coding:utf-8 -*-

import os
import time
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import RandomSampler
from torch.utils.data import DataLoader, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

from qdnet.conf.config import load_yaml
from qdnet.optimizer.optimizer import GradualWarmupSchedulerV2
from qdnet.dataset.dataset import get_df, QDDataset
from qdnet.dataaug.dataaug import get_transforms
from qdnet.models.effnet import Effnet
from qdnet.models.resnest import Resnest
from qdnet.models.se_resnext import SeResnext
from qdnet.models.resnet import Resnet
from qdnet.models.mobilenetv2 import Mobilenet
from qdnet.loss.loss import Loss
from qdnet.conf.constant import Constant


parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--config_path', help='config file path')
parser.add_argument('--n_splits', help='n_splits', type=int)
args = parser.parse_args()
config = load_yaml(args.config_path, args)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def val_epoch(model, loader, mel_idx, get_output=False):

    model.eval()
    val_loss = []
    LOGITS = []
    PROBS = []
    TARGETS = []
    test_acc=0
    with torch.no_grad():
        for (data, target) in tqdm(loader):
            
            data, target = data.to(device), target.to(device)
            logits = torch.zeros((data.shape[0], int(config["out_dim"]))).to(device)  
            probs = torch.zeros((data.shape[0], int(config["out_dim"]))).to(device)  

            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())

            loss = Loss(out_dim=int(config["out_dim"]), loss_type=config["loss_type"])(model, data, target, mixup_cutmix=False)
            val_loss.append(loss.detach().cpu().numpy())

            outputs = model(data)
            ti_acc, = accuracy(outputs, target)
            test_acc += float(ti_acc[0])

    val_loss = np.mean(val_loss)
    LOGITS = torch.cat(LOGITS).numpy()
    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()

    # if get_output:
    #     return LOGITS, PROBS
    # else:
    acc = test_acc / len(loader)
    auc = 0
    return val_loss, acc, auc



def main():

    df, df_val, df_test, mel_idx = get_df( config["data_dir"], config["auc_index"]  )

    _, transforms_val = get_transforms(int(config["image_size"]))   

    LOGITS = []
    PROBS = []
    dfs = []

    df_valid = df_test

    dataset_valid = QDDataset(df_valid, 'valid', transform=transforms_val)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=int(config["batch_size"]), num_workers=int(config["num_workers"]))

    if config["eval"] == 'best':
        model_file = os.path.join(config["model_dir"], f'best.pth')
    if config["eval"] == 'final':
        model_file = os.path.join(config["model_dir"], f'final.pth')

    model = ModelClass(
        enet_type = config["enet_type"],
        out_dim = int(config["out_dim"]),
        drop_nums = int(config["drop_nums"]),
        metric_strategy = config["metric_strategy"]
    )
    model = model.to(device)

    try:  # single GPU model_file
        model.load_state_dict(torch.load(model_file), strict=True)
    except:  # multi GPU model_file
        state_dict = torch.load(model_file)
        state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
        model.load_state_dict(state_dict, strict=True)

    if len(os.environ['CUDA_VISIBLE_DEVICES']) > 1:
        model = torch.nn.DataParallel(model)

    model.eval()

    val_loss, acc, auc = val_epoch(model, valid_loader, mel_idx, get_output=True)
    print(f" valid loss: {(val_loss):.5f}, acc: {(acc):.4f}.")
    # LOGITS.append(this_LOGITS)
    # PROBS.append(this_PROBS)
    # dfs.append(df_valid)
    #
    # dfs = pd.concat(dfs).reset_index(drop=True)
    # dfs['pred'] = np.concatenate(PROBS).squeeze()[:, mel_idx]

    # auc_all_raw = roc_auc_score(dfs['target'] == mel_idx, dfs['pred'])

    # dfs2 = dfs.copy()
    # for i in range(args.n_splits):
    #     dfs2.loc[dfs2['fold'] == i, 'pred'] = dfs2.loc[dfs2['fold'] == i, 'pred'].rank(pct=True)
    # auc_all_rank = roc_auc_score(dfs2['target'] == mel_idx, dfs2['pred'])
    
    # content = f'Eval {config["eval"]}:\nauc_all_raw : {auc_all_raw:.5f}\nauc_all_rank : {auc_all_rank:.5f}\n'
    # print(content)
    # with open(os.path.join(config["log_dir"], f'log.txt'), 'a') as appender:
    #     appender.write(content + '\n')

    # np.save(os.path.join(config["oof_dir"], f'{config["eval"]}_oof.npy'), dfs['pred'].values)


if __name__ == '__main__':

    os.makedirs(config["oof_dir"], exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = config["CUDA_VISIBLE_DEVICES"]

    if config["enet_type"] in Constant.RESNET_LIST:
        ModelClass = Resnet
    elif config["enet_type"] in Constant.MOBILENET_LIST:
        ModelClass = Mobilenet
    else:
        raise NotImplementedError()

    DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

    device = torch.device('cuda')
    criterion = nn.CrossEntropyLoss()

    main()
