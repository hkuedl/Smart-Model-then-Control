#%%
import numpy as np
import math
import pandas as pd
import os
import random
import copy
import torch
import time
def set_seed(seed):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False # train speed is slower after enabling this opts.

    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    torch.use_deterministic_algorithms(True)

set_seed(20)

import torch.nn as nn
import torch.optim as optim
device = torch.device("cuda:2")
import DF_Model_Func
import matplotlib.pyplot as plt

def to_np(x):
    return x.cpu().detach().numpy()

nn_zone = 90
T_Fre = 4 
Train_s = T_Fre*24*(31+28+31+30+31) 
Train_e = Train_s + T_Fre*24*15
Train_s2 = Train_e + T_Fre*24*15
Train_e2 = Train_s2 + T_Fre*24*15
train_period = T_Fre*24  
predict_period = T_Fre*4  

SS_X,SS_P,SS_Y,S_X_tr,S_P_tr,S_Y_tr,S_X_te,S_P_te,S_Y_te,X_tr,X_te,Y_tr,Y_te \
    = DF_Model_Func.data_input(T_Fre,Train_s,Train_e,Train_s2,Train_e2,train_period,nn_zone)

SS_Xmax,SS_Xmin = [SS_X.data_max_[i] for i in range(S_X_tr.shape[1])],[SS_X.data_min_[i] for i in range(S_X_tr.shape[1])]

rows_to_exclude = [99,108,117,126,135,144,153,162,171,180]
mask = np.ones(1+2*nn_zone, dtype=bool)
mask[rows_to_exclude] = False
Seq_disturb = ['out','rad','occ']
P_trte_acc_sum = np.zeros((1+nn_zone*2,6))
for Seq_dis_i in range(0,1+nn_zone*2):    #out,rad,occ
    set_seed(20)


    Seq_tr = DF_Model_Func.LSTM_sequences(S_X_tr[:,Seq_dis_i:Seq_dis_i+1], train_period,predict_period)
    Seq_te = DF_Model_Func.LSTM_sequences(S_X_te[:,Seq_dis_i:Seq_dis_i+1], train_period,predict_period)

    P_epoch = 1000 #2000
    P_epoch_freq = int(P_epoch/10)
    P_nn_input,P_nn_hidden,P_nn_layer,P_nn_output = 1,20,3,predict_period
    P_batch_size = len(Seq_tr)
    P_batch_num = math.ceil(len(Seq_tr)/P_batch_size)

    P_model = DF_Model_Func.LSTM(P_nn_input, P_nn_hidden, P_nn_layer, P_nn_output).to(device)

    P_optimizer = optim.Adam(P_model.parameters(),lr = 1e-3)
    P_lossfn = nn.MSELoss()

    for e in range(P_epoch):
        P_batch_list = list(range(len(Seq_tr)))
        for num in range(P_batch_num):
            P_batch_size_i = min(P_batch_size,len(P_batch_list))
            P_batch_list_i = random.sample(P_batch_list,P_batch_size_i)
            P_batch_list = [x for x in P_batch_list if x not in P_batch_list_i]
            P_optimizer.zero_grad()
            P_batch_data = torch.zeros((P_batch_size_i,train_period,1)).to(device)
            P_batch_label = torch.zeros((P_batch_size_i,predict_period)).to(device)
            for i in range(len(P_batch_list_i)):
                P_batch_data[i,:,:] = Seq_tr[P_batch_list_i[i]][0]
                P_batch_label[i,:] = Seq_tr[P_batch_list_i[i]][1].squeeze()
            P_batch_pred = P_model(P_batch_data)
            P_loss = P_lossfn(P_batch_pred,P_batch_label)
            P_loss.backward()
            P_optimizer.step()
        if e % P_epoch_freq == 0:
            with torch.no_grad():
                print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(e, (num+1) * P_batch_size, len(Seq_tr), P_loss.item()))
    
    P_test_data = torch.zeros((len(Seq_te),train_period,1)).to(device)
    for i in range(len(Seq_te)):
        P_test_data[i,:,:] = Seq_te[i][0]
    P_test_pred = to_np(P_model(P_test_data)).reshape(-1,1)
    P_test_huanyuan = P_test_pred*(SS_Xmax[Seq_dis_i]-SS_Xmin[Seq_dis_i])+SS_Xmin[Seq_dis_i]
    P_test_acc = DF_Model_Func.LSTM_acc(P_test_huanyuan,X_te[train_period:,Seq_dis_i:Seq_dis_i+1])

    P_train_data = torch.zeros((len(Seq_tr),train_period,1)).to(device)
    for i in range(len(Seq_tr)):
        P_train_data[i,:,:] = Seq_tr[i][0]
    P_train_pred = to_np(P_model(P_train_data)).reshape(-1,1)
    P_train_huanyuan = P_train_pred*(SS_Xmax[Seq_dis_i]-SS_Xmin[Seq_dis_i])+SS_Xmin[Seq_dis_i]
    P_train_acc = DF_Model_Func.LSTM_acc(P_train_huanyuan,X_tr[train_period:,Seq_dis_i:Seq_dis_i+1])

    P_trte_acc_sum[Seq_dis_i,:3],P_trte_acc_sum[Seq_dis_i,3:] = P_train_acc[:],P_test_acc[:]

    torch.save(P_model.state_dict(), 'Independent_'+str(nn_zone)+'zone_LSTM_'+str(Seq_dis_i)+'.pt')

print(P_trte_acc_sum[0,:])
print(np.mean(P_trte_acc_sum[1:1+nn_zone,:],axis = 0))
P_trte_acc_sum_data = P_trte_acc_sum[1+nn_zone:,:][mask[1+nn_zone:]]
print(np.mean(P_trte_acc_sum_data,axis = 0))

