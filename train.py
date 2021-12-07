import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *
from sklearn.metrics import mean_squared_error
from test import *
import torch.nn.functional as F
import numpy as np
from evaluate import get_best_performance_data, get_val_performance_data, get_full_err_scores
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from torch.utils.data import DataLoader, random_split, Subset
from scipy.stats import iqr
import gaussian_loss
import math
from timeit import default_timer as timer
from tqdm import tqdm

#def loss_func(y_pred, y_true):
#    loss = F.mse_loss(y_pred, y_true, reduction='mean')

#    return loss

# def loss_func(y_pred, y_true, var, alpha):
#     loss = gaussian_loss.gaussian_nll_loss(y_pred, y_true, var)
#     loss += alpha * torch.mean((torch.sqrt(var)-1.0) ** 2)
#     return loss

def loss_func_crossentropy(y_pred, y_true):
    crossentropy = torch.nn.CrossEntropyLoss()
    
    return crossentropy(y_pred, y_true)


def train(model = None, save_path = '', config={},  train_dataloader=None, val_dataloader=None, feature_map={}, test_dataloader=None, test_dataset=None, dataset_name='swat', train_dataset=None):

    seed = config['seed']

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=config['decay'])

    now = time.time()
    
    train_loss_list = []
    cmp_loss_list = []

    device = get_device()


    acu_loss = 0
    min_loss = 1e+8
    min_f1 = 0
    min_pre = 0
    best_prec = 0

    i = 0
    epoch = config['epoch']
    early_stop_win = 15

    model.train()

    log_interval = 1000
    stop_improve_count = 0

    dataloader = train_dataloader

    # Training iterations
    for i_epoch in range(epoch):
        start = timer()
        print("Running epoch ", i_epoch)
        acu_loss = 0
        model.train()

        for x, labels, attack_labels, edge_index in tqdm(dataloader):
            _start = time.time()

            x, labels, edge_index = [item.float().to(device) for item in [x, labels, edge_index]]

            optimizer.zero_grad()
            out = model(x, edge_index).float().to(device)
            
#             var = torch.abs(out[:,:,1])
#             mean = out[:,:,0]
#             loss = loss_func(mean, labels, var, config['alpha'])
            attack_labels = attack_labels.long().to(device)
            # out = (batch_size , 2)
            loss = loss_func_crossentropy(out, attack_labels)
    
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss_list.append(loss.item())
            acu_loss += loss.item()
                
            i += 1


        # each epoch
        print('epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f})'.format(
                        i_epoch, epoch, 
                        acu_loss/len(dataloader), acu_loss), flush=True
            )

        # use val dataset to judge
        if val_dataloader is not None:

            val_loss, val_result = test(model, val_dataloader)

            if val_loss < min_loss:
                torch.save(model.state_dict(), save_path)

                min_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1


            if stop_improve_count >= early_stop_win:
                break

        else:
            if acu_loss < min_loss :
                torch.save(model.state_dict(), save_path)
                min_loss = acu_loss

        end = timer()
        print("Time used by epoch : ", (end - start))

    return train_loss_list
