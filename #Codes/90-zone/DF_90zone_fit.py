#%%
import numpy as np
import math
import pandas as pd
import os
import random
import torch
import time
#% 当对权重有非负的限制时，训练神经网络
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
device = torch.device("cuda:2")
import DF_Model_Func
import matplotlib.pyplot as plt

############
def to_np(x):
    return x.cpu().detach().numpy()

nn_zone = 90 
T_Fre = 4  
Train_s = T_Fre*24*(31+28+31+30+31)
Train_e = Train_s + T_Fre*24*8

Train_s2 = Train_e
Train_e2 = Train_s2 + T_Fre*24*8

train_period = T_Fre*24

predict_period = T_Fre*4

SS_X,SS_P,SS_Y,S_X_tr,S_P_tr,S_Y_tr,S_X_te,S_P_te,S_Y_te,X_tr,X_te,Y_tr,Y_te \
    = DF_Model_Func.data_input(T_Fre,Train_s,Train_e,Train_s2,Train_e2,train_period,nn_zone)

#%%
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor as LOF

M_train_data,M_train_label = DF_Model_Func.MZ_model_data(S_X_tr,S_P_tr,S_Y_tr, train_period, predict_period)
M_test_data,M_test_label = DF_Model_Func.MZ_model_data(S_X_te,S_P_te,S_Y_te, train_period, predict_period)

C_period_train,C_period_test = int((Train_e-Train_s)/predict_period),int((Train_e2-Train_s2)/predict_period)
Daily_opt = int(24*T_Fre/predict_period)
Daily_days = int((Train_e-Train_s)/(24*T_Fre))

AB_size = 100 + 200

def ICNN_tr(CVXNN,nn_zone,m_state_dict, Map_fea_tr, Map_lab_tr_01,Map_epoch, hidden, predict_period):
    if CVXNN == 'no_dim':
        Map_model = DF_Model_Func.CvxModel(n_feature = Map_fea_tr.shape[1], n_hidden = hidden, n_output=1).to(device)
        Map_cons = DF_Model_Func.weightConstraint()
    elif CVXNN == 'dim':
        Map_model = DF_Model_Func.MZ_CvxModel(n_zone = nn_zone, n_feature = int(Map_fea_tr.shape[1]/nn_zone), n_hidden = hidden, n_output=1, predict_period = predict_period).to(device)
        Map_cons = DF_Model_Func.weightConstraint()
    elif CVXNN == 'fnn_nodim':
        Map_model = DF_Model_Func.FNNModel(n_feature = Map_fea_tr.shape[1], n_hidden = hidden, n_output=1).to(device)
    elif CVXNN == 'fnn_dim':
        Map_model = DF_Model_Func.MZ_FNNModel(n_zone = nn_zone, n_feature = int(Map_fea_tr.shape[1]/nn_zone), n_hidden = hidden, n_output=1, predict_period = predict_period).to(device)
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
            if CVXNN == 'no_dim':
                _,_,_,Map_pred = Map_model(Map_batch_fea)
            elif CVXNN == 'dim':
                Map_pred,comp = Map_model(Map_batch_fea)
            elif CVXNN == 'fnn_nodim':
                Map_pred = Map_model(Map_batch_fea)
            elif CVXNN == 'fnn_dim':
                Map_pred,comp = Map_model(Map_batch_fea)
            
            Map_pred.to(device)
            if CVXNN == 'no_dim' or CVXNN == 'fnn_nodim':
                Map_loss = Map_lossfn(Map_pred, Map_batch_lab)
            elif CVXNN == 'dim' or CVXNN == 'fnn_dim':
                Map_loss1 = Map_lossfn(Map_pred, Map_batch_lab)
                Map_loss = Map_loss1 + 10*comp[0] + 10*comp[1]

            Map_loss.backward()
            Map_optimizer.step()
            if CVXNN == 'dim' or CVXNN == 'no_dim':
                Map_model._modules['input_layer'].apply(Map_cons)
                Map_model._modules['hidden_layer1'].apply(Map_cons)
                Map_model._modules['hidden_layer2'].apply(Map_cons)
                Map_model._modules['output_layer'].apply(Map_cons)
        
        Map_loss_iter.append(Map_loss.item())
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

for CVXNN in ['dim','no_dim']:     #'no_dim', 'dim', 'fnn_dim', 'fnn_nodim'
    ICNN_time = []
    Err_all = np.zeros((int(Daily_days*Daily_opt),6))
    Weights_all = np.zeros((int(Daily_days*Daily_opt),nn_zone+1))
    for Map_day in range(0, Daily_days):
        for Map_prd in range(0, Daily_opt):
            Map_fea = np.zeros((AB_size,nn_zone*predict_period))
            for ii in range(nn_zone):
                i_st = Map_day*((predict_period+1)*Daily_opt)
                T_feaii = np.array(pd.read_excel('Sam_'+str(nn_zone)+'zone_opt.xlsx',sheet_name='tem_tr_'+str(ii)),dtype=float)
                T_feaii_new = np.array(pd.read_excel('Sam_'+str(nn_zone)+'zone_opt_new.xlsx',sheet_name='tem_tr_'+str(ii)),dtype=float)
                
                Map_fea[:100,ii*predict_period:(ii+1)*predict_period] = T_feaii[:100,i_st+(predict_period+1)*Map_prd+1:i_st+(predict_period+1)*(Map_prd+1)]
                Map_fea[100:AB_size,ii*predict_period:(ii+1)*predict_period] = T_feaii_new[:200,i_st+(predict_period+1)*Map_prd+1:i_st+(predict_period+1)*(Map_prd+1)]
                
            T_optlab = np.array(pd.read_excel('Sam_'+str(nn_zone)+'zone_opt_label.xlsx',sheet_name='label'))
            T_optlab_new = np.array(pd.read_excel('Sam_'+str(nn_zone)+'zone_opt_label_new.xlsx',sheet_name='label'))

            Map_lab = np.vstack((T_optlab[Map_day*Daily_opt+Map_prd,:100].reshape((-1,1)),T_optlab_new[Map_day*Daily_opt+Map_prd,:200].reshape((-1,1))))

            del T_feaii
            del T_feaii_new
            
            clf = LOF(n_neighbors = 10)  #1,10
            predict = clf.fit_predict(Map_fea)
            Map_fea = Map_fea[np.where(predict == 1),:][0]
            Map_lab = Map_lab[np.where(predict == 1),:][0]

            SS_fea = MinMaxScaler().fit(Map_fea)
            Map_fea_01 = SS_fea.transform(Map_fea)
            SS_lab = MinMaxScaler().fit(Map_lab)
            Map_lab_01 = SS_lab.transform(Map_lab)
            Map_fea_tr, Map_fea_te, Map_lab_tr_01, Map_lab_te_01 = train_test_split(Map_fea_01, Map_lab_01, test_size=0.2, random_state = 20)
            
            hidden = 32
            if CVXNN == 'no_dim':
                Map_model_ini = DF_Model_Func.CvxModel(n_feature = Map_fea_tr.shape[1], n_hidden = hidden, n_output=1).to(device)
            elif CVXNN == 'dim':
                Map_model_ini = DF_Model_Func.MZ_CvxModel(n_zone = nn_zone, n_feature = int(Map_fea_tr.shape[1]/nn_zone), n_hidden = hidden, n_output=1, predict_period = predict_period).to(device)
            elif CVXNN == 'fnn_nodim':
                Map_model_ini = DF_Model_Func.FNNModel(n_feature = Map_fea_tr.shape[1], n_hidden = hidden, n_output=1).to(device)
            elif CVXNN == 'fnn_dim':
                Map_model_ini = DF_Model_Func.MZ_FNNModel(n_zone = nn_zone, n_feature = int(Map_fea_tr.shape[1]/nn_zone), n_hidden = hidden, n_output=1, predict_period = predict_period).to(device)

            Map_epoch = 5000
            time1 = time.time()
            Map_model,Map_loss_iter = ICNN_tr(CVXNN,nn_zone,Map_model_ini.state_dict(), Map_fea_tr, Map_lab_tr_01, Map_epoch, hidden, predict_period)
            time2 = time.time()
            ICNN_time.append(time2-time1)
            if CVXNN == 'no_dim':
                _,_,_,Map_TR_pred_01 = Map_model(torch.tensor(Map_fea_tr, dtype=torch.float32).to(device))
                _,_,_,Map_TE_pred_01 = Map_model(torch.tensor(Map_fea_te, dtype=torch.float32).to(device))
            elif CVXNN == 'dim':
                Map_TR_pred_01,comp_tr = Map_model(torch.tensor(Map_fea_tr, dtype=torch.float32).to(device))
                Map_TE_pred_01,comp_te = Map_model(torch.tensor(Map_fea_te, dtype=torch.float32).to(device))
            elif CVXNN == 'fnn_nodim':
                Map_TR_pred_01 = Map_model(torch.tensor(Map_fea_tr, dtype=torch.float32).to(device))
                Map_TE_pred_01 = Map_model(torch.tensor(Map_fea_te, dtype=torch.float32).to(device))
            elif CVXNN == 'fnn_dim':
                Map_TR_pred_01,comp_tr = Map_model(torch.tensor(Map_fea_tr, dtype=torch.float32).to(device))
                Map_TE_pred_01,comp_te = Map_model(torch.tensor(Map_fea_te, dtype=torch.float32).to(device))
            Map_TR_pred = SS_lab.inverse_transform(to_np(Map_TR_pred_01))
            Map_TE_pred = SS_lab.inverse_transform(to_np(Map_TE_pred_01))
            Map_lab_tr = SS_lab.inverse_transform(Map_lab_tr_01)
            Map_lab_te = SS_lab.inverse_transform(Map_lab_te_01)
            Map_err_tr = Map_acc(Map_lab_tr,Map_TR_pred)
            Map_err_te = Map_acc(Map_lab_te,Map_TE_pred)

            Err_all[Map_day*Daily_opt+Map_prd,0:3],Err_all[Map_day*Daily_opt+Map_prd,3:6] = Map_err_tr,Map_err_te
            IC_input_01 = np.zeros((2,nn_zone*predict_period))
            for i in range(nn_zone*predict_period):
                IC_input_01[0,i],IC_input_01[1,i] = SS_fea.data_min_[i],SS_fea.data_max_[i]
            np.save('Result_ICNN_'+str(nn_zone)+'zone/IC_input_'+str(nn_zone)+'zone_'+str(Map_day)+'_'+str(Map_prd)+'.npy',IC_input_01)
            if CVXNN == 'dim':
                Weights_all[Map_day*Daily_opt+Map_prd,:-1] = to_np(Map_model.w)[:,0]
                Weights_all[Map_day*Daily_opt+Map_prd,-1] = sum(to_np(Map_model.w))
                torch.save(Map_model.state_dict(), 'Result_ICNN_'+str(nn_zone)+'zone/'+str(nn_zone)+'zone_ICNN_model_'+str(Map_day)+'_'+str(Map_prd)+'.pt')
            if CVXNN == 'no_dim':
                torch.save(Map_model.state_dict(), 'Result_ICNN_'+str(nn_zone)+'zone/'+str(nn_zone)+'zone_ICNN_nodim_model_'+str(Map_day)+'_'+str(Map_prd)+'.pt')
            if CVXNN == 'fnn_dim':
                Weights_all[Map_day*Daily_opt+Map_prd,:-1] = to_np(Map_model.w)[:,0]
                Weights_all[Map_day*Daily_opt+Map_prd,-1] = sum(to_np(Map_model.w))
                torch.save(Map_model.state_dict(), 'Result_FNN_'+str(nn_zone)+'zone/'+str(nn_zone)+'zone_FNN_model_'+str(Map_day)+'_'+str(Map_prd)+'.pt')
            if CVXNN == 'fnn_nodim':
                torch.save(Map_model.state_dict(), 'Result_FNN_'+str(nn_zone)+'zone/'+str(nn_zone)+'zone_FNN_nodim_model_'+str(Map_day)+'_'+str(Map_prd)+'.pt')

    print(np.mean(np.array(ICNN_time)))
    print(np.std(np.array(ICNN_time)))  
