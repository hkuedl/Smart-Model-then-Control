#%%
import numpy as np
import math
import pandas as pd
import os
import random
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
import cvxpy as cp
import DF_Model_Func

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


#%% single LSTM

set_seed(20)
Seq_disturb = ['out','rad','occ']
Seq_dis_i = 0    #out,rad,occ

Seq_tr = DF_Model_Func.LSTM_sequences(S_X_tr[:,Seq_dis_i:Seq_dis_i+1], train_period,predict_period)
Seq_te = DF_Model_Func.LSTM_sequences(S_X_te[:,Seq_dis_i:Seq_dis_i+1], train_period,predict_period)

P_epoch = 2000
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

torch.save(P_model.state_dict(), 'Independent_2zone_LSTM_'+str(Seq_dis_i)+'.pt')


#%%
set_seed(20)
M_train_data,M_train_label = DF_Model_Func.model_data(S_X_tr,S_P_tr,S_Y_tr, train_period, predict_period)
M_test_data,M_test_label = DF_Model_Func.model_data(S_X_te,S_P_te,S_Y_te, train_period, predict_period)
M_epoch = 4000
M_epoch_freq = int(M_epoch/10)
M_nn_input,M_nn_hidden,M_nn_output = M_train_data.shape[-1]-2,20,1
M_layers = 2
M_batch_size = M_train_data.shape[1]
M_batch_num = math.ceil(M_train_data.shape[1]/M_batch_size)

M_model = DF_Model_Func.model_FNN(M_nn_input, M_nn_hidden, M_nn_output, M_layers).to(device)
M_optimizer = optim.Adam(M_model.parameters(),lr = 1e-2)
M_lossfn = nn.MSELoss()

time1 = time.time()
for m in range(M_epoch):
    M_batch_list = list(range(M_train_data.shape[1]))
    for num in range(M_batch_num):
        M_batch_size_i = min(M_batch_size,len(M_batch_list))
        M_batch_list_i = random.sample(M_batch_list,M_batch_size_i)
        M_batch_list = [x for x in M_batch_list if x not in M_batch_list_i]
        M_optimizer.zero_grad()
        M_batch_data = M_train_data[:-1,M_batch_list_i,:]
        M_batch_label = M_train_label[:,M_batch_list_i]
        M_batch_pred = M_model(M_batch_data)
        M_batch_pred.to(device)
        M_loss = M_lossfn(M_batch_pred,M_batch_label)
        M_loss.backward()
        M_optimizer.step()
    if m % M_epoch_freq == 0:
        with torch.no_grad():
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(m, (num+1) * M_batch_size, M_train_data.shape[1], M_loss.item()))

time2 = time.time()

print(time2-time1)

M_test_pred = to_np(M_model(M_test_data[:-1,:,:]))
M_test_pred_1 = M_test_pred.reshape((M_test_pred.shape[0]*M_test_pred.shape[1], 1), order='F')
M_test_huanyuan = SS_Y.inverse_transform(M_test_pred_1)
M_test_acc = DF_Model_Func.LSTM_acc(M_test_huanyuan,Y_te[train_period:,0:1])

M_train_pred = to_np(M_model(M_train_data[:-1,:,:]))
M_train_pred_1 = M_train_pred.reshape((M_train_pred.shape[0]*M_train_pred.shape[1], 1), order='F')
M_train_huanyuan = SS_Y.inverse_transform(M_train_pred_1)
M_train_acc = DF_Model_Func.LSTM_acc(M_train_huanyuan,Y_tr[train_period:,0:1])

torch.save(M_model.state_dict(), 'Independent_2zone_Model.pt')


#%%
Daily_opt = int(24*T_Fre/predict_period)
Daily_days = int((Train_e-Train_s)/(24*T_Fre))
Daily_days_te = int((Train_e2-Train_s2)/(24*T_Fre))
C_period_train,C_period_test = int((Train_e-Train_s)/predict_period),int((Train_e2-Train_s2)/predict_period)

P_nn_input,P_nn_hidden,P_nn_layer,P_nn_output = 1,20,3,predict_period
P_models = []
for i in range(3):
    P_models.append(DF_Model_Func.LSTM(P_nn_input, P_nn_hidden, P_nn_layer, P_nn_output).to(device))
    m_state_dict = torch.load('Independent_2zone_LSTM_'+str(i)+'.pt')
    P_models[i].load_state_dict(m_state_dict)

M_train_data,M_train_label = DF_Model_Func.model_data(S_X_tr,S_P_tr,S_Y_tr, train_period, predict_period)
M_test_data,M_test_label = DF_Model_Func.model_data(S_X_te,S_P_te,S_Y_te, train_period, predict_period)
M_nn_input,M_nn_hidden,M_nn_output = M_train_data.shape[-1]-2,20,1
M_layers = 2
M_model = DF_Model_Func.model_FNN(M_nn_input, M_nn_hidden, M_nn_output, M_layers).to(device)
m_state_dict = torch.load('Independent_2zone_Model.pt')
M_model.load_state_dict(m_state_dict)

tem_base = np.zeros((Daily_days,24))
for i in range(tem_base.shape[0]):
    tem_base[i,:] = Y_tr[::T_Fre,0][i*24:(i+1)*24]
tem_base_mean = np.round(np.mean(tem_base,axis = 0),1)
c_0_tem_comfort = [1.4,1.4,1.4,1.4,1.4,1.4,1.2,1.2,1.2,1.2,0.8,0.8,0.8,1.0,1.0,1.0,1.0,1.0,1.2,1.2,1.2,1.4,1.4,1.4]
c_0_tem = np.zeros((24,2))
for i in range(24):
    c_0_tem[i,0] = tem_base_mean[i]-c_0_tem_comfort[i]*1.0
    c_0_tem[i,1] = tem_base_mean[i]+c_0_tem_comfort[i]*1.0

C_price,C_upper_tem,C_lower_tem,C_upper_q,C_lower_q = DF_Model_Func.opt_settings(nn_zone, c_0_tem, SS_P,SS_Y, Train_s,Train_e,Train_s2,Train_e2,predict_period,T_Fre)
Cr_obj = np.zeros((C_period_train+C_period_test,3))
Cr_q = np.zeros((C_period_train+C_period_test,predict_period))
Cr_tem = np.zeros((C_period_train+C_period_test,predict_period+1))

Seq_input_tr,Seq_input_te = [],[]
for i in range(S_X_tr.shape[1]):
    Seq_input_tr.append(DF_Model_Func.LSTM_sequences(S_X_tr[:,i:i+1], train_period,predict_period))
    Seq_input_te.append(DF_Model_Func.LSTM_sequences(S_X_te[:,i:i+1], train_period,predict_period))

Cr_q_hy = np.zeros((Daily_days+Daily_days_te,Daily_opt*predict_period))
Cr_tem_hy = np.zeros((Daily_days+Daily_days_te,Daily_opt*(predict_period+1)))

Opt_time = []
for c_day_index in range(0, Daily_days+Daily_days_te):
    for c_prd_index in range(Daily_opt):
        c_i_index = c_day_index*Daily_opt + c_prd_index
        c_v_tem,c_v_q,c_v_tem_u,c_p_A,c_p_B,c_p_F,c_prob,c_layer,c_obj1,c_obj2 = DF_Model_Func.opt_problem(C_price,C_upper_tem,C_lower_tem,C_upper_q,C_lower_q,SS_P, SS_Y, predict_period,T_Fre, c_i_index)
        c_p_A.value = M_model.a.item()
        c_p_B.value = M_model.b.item()
    
        Seq_output = torch.zeros((predict_period,3)).to(device)
        for i in range(3):
            if c_day_index >= Daily_days:
                Seq_input_i = Seq_input_te[i][c_i_index-Daily_days*Daily_opt][0].reshape(1,train_period,1)
            else:
                Seq_input_i = Seq_input_tr[i][c_i_index][0].reshape(1,train_period,1)
            Seq_output[:,i] = P_models[i](Seq_input_i.to(torch.float32))
        c_p_F.value = to_np(M_model.net(Seq_output))[:,0]
        
        time3 = time.time()
        c_prob.solve(solver = cp.GUROBI,verbose=False)
        time4 = time.time()
        Opt_time.append(time4-time3)
        Cr_obj[c_i_index,:] = [c_prob.value,c_obj1.value,c_obj2.value]
        Cr_q[c_i_index,:] = c_v_q.value
        Cr_tem[c_i_index,:] = c_v_tem.value
        Cr_q_hy[c_day_index,predict_period*c_prd_index:predict_period*(c_prd_index+1)] = SS_P.inverse_transform(c_v_q.value.reshape(-1,1)).ravel()
        Cr_tem_hy[c_day_index,(predict_period+1)*c_prd_index:(predict_period+1)*(c_prd_index+1)] = SS_Y.inverse_transform(c_v_tem.value.reshape(-1,1)).ravel()

Cr_obj_trte = [sum(Cr_obj[:C_period_train,0]),sum(Cr_obj[C_period_train:,0])]
Cr_obj_trte1 = [sum(Cr_obj[:C_period_train,1]),sum(Cr_obj[C_period_train:,1])]
Cr_obj_trte2 = [sum(Cr_obj[:C_period_train,2]),sum(Cr_obj[C_period_train:,2])]


print(np.mean(np.array(Opt_time)))
print(np.std(np.array(Opt_time)))

writer = pd.ExcelWriter('Ind_2zone_opt.xlsx')
to_tem_limit = pd.DataFrame(c_0_tem)
to_tem_limit.to_excel(writer,sheet_name='tem_limits',index=False)

to_Cr_tem = pd.DataFrame(Cr_tem_hy)
to_Cr_tem.to_excel(writer,sheet_name='tem_trte',index=False)

to_Cr_q = pd.DataFrame(Cr_q_hy)
to_Cr_q.to_excel(writer,sheet_name='q_trte',index=False)
writer.close()
