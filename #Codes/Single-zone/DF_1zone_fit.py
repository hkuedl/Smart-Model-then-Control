#%%
import numpy as np
import math
import pandas as pd
import os
import random
import copy
import torch
import time
from sklearn.model_selection import train_test_split
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
device = torch.device("cuda:1")
import DF_Model_Func
import matplotlib.pyplot as plt

############
def to_np(x):
    return x.cpu().detach().numpy()

nn_zone = 1
T_Fre = 4   
Train_s = T_Fre*24*(31+28+31+30+31) 
Train_e = Train_s + T_Fre*24*(30)
Train_s2 = Train_e
Train_e2 = Train_s2 + T_Fre*24*31
train_period = T_Fre*24  
predict_period = T_Fre*8  

SS_X,SS_P,SS_Y,S_X_tr,S_P_tr,S_Y_tr,S_X_te,S_P_te,S_Y_te,X_tr,X_te,Y_tr,Y_te \
    = DF_Model_Func.data_input(T_Fre,Train_s,Train_e,Train_s2,Train_e2,train_period,nn_zone)

SS_Xmax,SS_Xmin = [SS_X.data_max_[i] for i in range(3)],[SS_X.data_min_[i] for i in range(3)]
SS_Pmax,SS_Pmin,SS_Ymax,SS_Ymin = SS_P.data_max_[0],SS_P.data_min_[0],SS_Y.data_max_[0],SS_Y.data_min_[0]


#%% 
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor as LOF

M_train_data,M_train_label = DF_Model_Func.model_data(S_X_tr,S_P_tr,S_Y_tr, train_period, predict_period)
M_test_data,M_test_label = DF_Model_Func.model_data(S_X_te,S_P_te,S_Y_te, train_period, predict_period)

C_period_train,C_period_test = int((Train_e-Train_s)/predict_period),int((Train_e2-Train_s2)/predict_period)
Daily_opt = int(24*T_Fre/predict_period)
Daily_days = int((Train_e-Train_s)/(24*T_Fre))
AB_size = 1000

def ICNN_tr(CVXNN,m_state_dict, Map_fea_tr, Map_lab_tr_01,Map_epoch):
    if CVXNN == 1:
        Map_model = DF_Model_Func.CvxModel(n_feature = Map_fea_tr.shape[1], n_hidden = 64, n_output=1).to(device)
        Map_cons = DF_Model_Func.weightConstraint()
    else:
        Map_model = DF_Model_Func.FNNModel(n_feature = Map_fea_tr.shape[1], n_hidden = 64, n_output=1).to(device)
    Map_model.load_state_dict(m_state_dict)
    Map_lossfn = torch.nn.MSELoss()
    Map_lr = 1e-3
    Map_optimizer = torch.optim.Adam(Map_model.parameters(), lr = Map_lr)
    Map_epoch_freq = int(Map_epoch/5)
    Map_batch_size = Map_fea_tr.shape[0]
    Map_batch_num = math.ceil(Map_fea_tr.shape[0]/Map_batch_size)
    Map_loss_iter = []
    for epoch in range(Map_epoch):
        Map_batch_list = list(range(Map_fea_tr.shape[0]))
        for num in range(Map_batch_num):
            Map_batch_size_i = min(Map_batch_size,len(Map_batch_list))
            Map_batch_list_i = random.sample(Map_batch_list,Map_batch_size_i)
            Map_batch_list = [x for x in Map_batch_list if x not in Map_batch_list_i]
            Map_optimizer.zero_grad()
            Map_batch_fea = torch.tensor(Map_fea_tr[Map_batch_list_i,:], dtype=torch.float32).to(device)
            Map_batch_lab = torch.tensor(Map_lab_tr_01[Map_batch_list_i,:], dtype=torch.float32).to(device)
            if CVXNN == 1:
                _,_,_,Map_pred = Map_model(Map_batch_fea)
            else:
                Map_pred = Map_model(Map_batch_fea)
            
            Map_pred.to(device)
            Map_loss = Map_lossfn(Map_pred, Map_batch_lab)

            Map_loss.backward()
            Map_optimizer.step()
            if CVXNN == 1:
                Map_model._modules['input_layer'].apply(Map_cons)
                Map_model._modules['hidden_layer1'].apply(Map_cons)
                Map_model._modules['hidden_layer2'].apply(Map_cons)
                Map_model._modules['output_layer'].apply(Map_cons)
        
        Map_loss_iter.append(Map_loss.item())
        if epoch % Map_epoch_freq == 0:
            with torch.no_grad():
                print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, (num+1)*Map_batch_size, Map_fea_tr.shape[0], Map_loss_iter[epoch]))
    return Map_model,Map_loss_iter
###test performance
from sklearn.metrics import r2_score
def Map_acc(data_true,data_pred):
    T_len = data_true.shape[0]
    Err_tr = np.abs(data_pred[:,0] - data_true[:,0])
    Err_tr1 = math.sqrt(sum(Err_tr[i]**2/(T_len) for i in range(T_len)))  #RMSE
    Err_tr2 = sum(Err_tr[i] for i in range(T_len))/(T_len)  #MAE
    Err_tr3 = r2_score(data_true, data_pred)  
    ERR2 = [Err_tr1,Err_tr2,Err_tr3]
    return ERR2

Err_all = np.zeros((int(Daily_days*Daily_opt),6))

ICNN_time = []

for Map_day in range(0, Daily_days):
    for Map_prd in range(0, Daily_opt):
        T_optfeas0 = np.array(pd.read_excel('Sam_2zone_opt.xlsx',sheet_name='tem_tr_'+str(Map_day)),dtype=float)
        T_optfeas = T_optfeas0[:,(predict_period+1)*Map_prd+1:(predict_period+1)*(Map_prd+1)]
        T_optlab = np.array(pd.read_excel('Sam_2zone_opt_label.xlsx',sheet_name='label'))

        Map_fea = copy.deepcopy(T_optfeas)
        Map_lab = T_optlab[Map_day*Daily_opt+Map_prd,:].reshape((-1,1))

        clf = LOF(n_neighbors = 10)  #1,10
        predict = clf.fit_predict(Map_fea)
        print(np.where(predict == -1))
        Map_fea = Map_fea[np.where(predict == 1),:][0]
        Map_lab = Map_lab[np.where(predict == 1),:][0]

        SS_fea = MinMaxScaler().fit(Map_fea)
        Map_fea = SS_fea.transform(Map_fea)

        SS_lab = MinMaxScaler().fit(Map_lab)
        Map_lab_01 = SS_lab.transform(Map_lab)
        Map_fea_tr, Map_fea_te, Map_lab_tr_01, Map_lab_te_01 = train_test_split(Map_fea, Map_lab_01, test_size=0.2, random_state = 20)

        CVXNN = 1
        if CVXNN == 1:
            Map_model_ini = DF_Model_Func.CvxModel(n_feature = Map_fea_tr.shape[1], n_hidden = 64, n_output=1).to(device)
            Map_cons = DF_Model_Func.weightConstraint()
        else:
            Map_model_ini = DF_Model_Func.FNNModel(n_feature = Map_fea_tr.shape[1], n_hidden = 64, n_output=1).to(device)

        Map_epoch = 10000
        time1 = time.time()
        Map_model,Map_loss_iter = ICNN_tr(CVXNN,Map_model_ini.state_dict(), Map_fea_tr, Map_lab_tr_01, Map_epoch)
        time2 = time.time()
        ICNN_time.append(time2-time1)
        
        if CVXNN == 1:
            _,_,_,Map_TR_pred_01 = Map_model(torch.tensor(Map_fea_tr, dtype=torch.float32).to(device))
            _,_,_,Map_TE_pred_01 = Map_model(torch.tensor(Map_fea_te, dtype=torch.float32).to(device))
        else:
            Map_TR_pred_01 = Map_model(torch.tensor(Map_fea_tr, dtype=torch.float32).to(device))
            Map_TE_pred_01 = Map_model(torch.tensor(Map_fea_te, dtype=torch.float32).to(device))
        Map_TR_pred = SS_lab.inverse_transform(to_np(Map_TR_pred_01))
        Map_TE_pred = SS_lab.inverse_transform(to_np(Map_TE_pred_01))
        Map_lab_tr = SS_lab.inverse_transform(Map_lab_tr_01)
        Map_lab_te = SS_lab.inverse_transform(Map_lab_te_01)
        Map_err_tr = Map_acc(Map_lab_tr,Map_TR_pred)
        Map_err_te = Map_acc(Map_lab_te,Map_TE_pred)

        Err_all[Map_day*Daily_opt+Map_prd,0:3],Err_all[Map_day*Daily_opt+Map_prd,3:6] = Map_err_tr,Map_err_te
        print([Map_err_tr[0],Map_err_tr[2],Map_err_te[0],Map_err_te[2]])

        IC_input_01 = np.zeros((2,predict_period))
        for i in range(predict_period):
            IC_input_01[0,i],IC_input_01[1,i] = SS_fea.data_min_[i],SS_fea.data_max_[i]
        if CVXNN == 1:
            torch.save(Map_model.state_dict(), '2zone_ICNN_model_'+str(Map_day)+'_'+str(Map_prd)+'.pt')
            np.save('IC_input_'+str(Map_day)+'_'+str(Map_prd)+'.npy',IC_input_01)
        else:
            torch.save(Map_model.state_dict(), '2zone_ICNN_model_'+str(Map_day)+'_'+str(Map_prd)+'.pt')

print(np.mean(np.array(ICNN_time)))
print(np.std(np.array(ICNN_time)))      
