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
import cvxpy as cp
import DF_Model_Func
import matplotlib.pyplot as plt

############
def to_np(x):
    return x.cpu().detach().numpy()
nn_zone = 22
T_Fre = 4
Train_s = T_Fre*24*(31+28+31+30+31)
Train_e = Train_s + T_Fre*24*(30)
Train_s2 = Train_e
Train_e2 = Train_s2 + T_Fre*24*31
train_period = T_Fre*24
predict_period = T_Fre*8

SS_X,SS_P,SS_Y,S_X_tr,S_P_tr,S_Y_tr,S_X_te,S_P_te,S_Y_te,X_tr,X_te,Y_tr,Y_te \
    = DF_Model_Func.data_input(T_Fre,Train_s,Train_e,Train_s2,Train_e2,train_period,nn_zone)

SS_Ymax,SS_Ymin = [SS_Y.data_max_[i] for i in range(nn_zone)],[SS_Y.data_min_[i] for i in range(nn_zone)]

Dis_tr_data,Dis_tr_label,Dis_te_data,Dis_te_label = [], [], [], []
for Seq_dis_i in range(S_X_tr.shape[-1]):
    Seq_tr = DF_Model_Func.LSTM_sequences(S_X_tr[:,Seq_dis_i:Seq_dis_i+1], train_period,predict_period)
    Seq_te = DF_Model_Func.LSTM_sequences(S_X_te[:,Seq_dis_i:Seq_dis_i+1], train_period,predict_period)

    Dis_tr_data_i = torch.zeros((len(Seq_tr),train_period,1)).to(device)
    Dis_tr_label_i = torch.zeros((len(Seq_tr),predict_period)).to(device)
    Dis_te_data_i = torch.zeros((len(Seq_te),train_period,1)).to(device)
    Dis_te_label_i = torch.zeros((len(Seq_te),predict_period)).to(device)
    for i in range(len(Seq_tr)):
        Dis_tr_data_i[i,:,:] = Seq_tr[i][0]
        Dis_tr_label_i[i,:] = Seq_tr[i][1].squeeze()
    for i in range(len(Seq_te)):
        Dis_te_data_i[i,:,:] = Seq_te[i][0]
        Dis_te_label_i[i,:] = Seq_te[i][1].squeeze()
    Dis_tr_data.append(Dis_tr_data_i)
    Dis_tr_label.append(Dis_tr_label_i)
    Dis_te_data.append(Dis_te_data_i)
    Dis_te_label.append(Dis_te_label_i)

Daily_opt = int(24*T_Fre/predict_period)
Daily_days = int((Train_e-Train_s)/(24*T_Fre))
Daily_days_te = int((Train_e2-Train_s2)/(24*T_Fre))
C_period_train,C_period_test = int((Train_e-Train_s)/predict_period),int((Train_e2-Train_s2)/predict_period)

P_nn_input,P_nn_hidden,P_nn_layer,P_nn_output = 1,20,3,predict_period
P_models = []
for i in range(S_X_tr.shape[-1]):
    m_state_dict = torch.load('Independent_22zone_LSTM_'+str(i)+'.pt')
    P_models.append(m_state_dict)
    

M_train_data,M_train_label = DF_Model_Func.MZ_model_data(S_X_tr,S_P_tr,S_Y_tr, train_period, predict_period)
M_test_data,M_test_label = DF_Model_Func.MZ_model_data(S_X_te,S_P_te,S_Y_te, train_period, predict_period)
M_nn_input,M_nn_hidden,M_nn_output = S_X_tr.shape[-1],20,S_Y_tr.shape[-1]
M_layers = 4
M_model = DF_Model_Func.MZ_model_FNN(M_nn_input, M_nn_hidden, M_nn_output, M_layers).to(device)
m_state_dict = torch.load('22zone_ini_model.pt')
M_model.load_state_dict(m_state_dict)

IC_map_all = []
IC_input_01_all = []
for d in range(Daily_days):
    for p in range(Daily_opt):
        IC_map = DF_Model_Func.MZ_CvxModel(n_zone = nn_zone, n_feature = predict_period, n_hidden = 64, n_output=1, predict_period = predict_period).to(device)
        m_state_dict = torch.load('22zone_ICNN_model_'+str(d)+'_'+str(p)+'.pt')
        IC_map.load_state_dict(m_state_dict)
        IC_map_all.append(IC_map)
        IC_input_01_all.append(np.load('IC_input_22zone_'+str(d)+'_'+str(p)+'.npy'))

class MZ_opt_joint_map_all(nn.Module):
    def __init__(self, P_models,M_model,IC_map_all):
        super(MZ_opt_joint_map_all, self).__init__()
        self.lstm_list = nn.ModuleList([DF_Model_Func.LSTM(1, 20, 3, 32) for _ in range(len(P_models))])
        for ii in range(len(P_models)):
            self.lstm_list[ii].load_state_dict(P_models[ii])
        self.model = M_model
        self.ICNN = IC_map_all
        
    def forward(self, Dis):
        D_pred = [self.lstm_list[i](Dis[i]).to(device) for i in range(len(Dis))]
        c_p_F = torch.zeros(D_pred[0].shape[0],D_pred[0].shape[1],22).to(device)
        for i in range(D_pred[0].shape[1]):  #32
            input = self.lstm_list[0](Dis[0])[:,i:i+1].to(device)
            for ii in range(1,45):
                input = torch.cat((input,self.lstm_list[ii](Dis[ii])[:,i:i+1]),dim = 1).to(device)
            c_p_F[:,i,:] = self.model.net(input)
        return D_pred,c_p_F

J_epoch = 100
J_epoch_freq = 1
J_train_samples = int((Train_e-Train_s)/predict_period)
J_batch_size = 18
J_batch_num = math.ceil(J_train_samples/J_batch_size)
J_model = MZ_opt_joint_map_all(P_models,M_model,IC_map_all).to(device)
for i in range(C_period_train):
    for param in J_model.ICNN[i].parameters():
        param.requires_grad = False
for i in range(S_X_tr.shape[-1]):
    if i in [1,3,4,5,6]:
        for param in J_model.lstm_list[i].parameters():
            param.requires_grad = False
    else:
        for param in J_model.lstm_list[i].parameters():
            param.requires_grad = True

J_optimizer = optim.Adam(J_model.parameters(),lr = 1e-4)

J_lossfn = nn.MSELoss()

J_loss = []
J_loss_obj = []
J_loss_dis = []
J_models = []

tem_base = np.zeros((Daily_days,24))
for i in range(tem_base.shape[0]):
    tem_base[i,:] = Y_tr[::T_Fre,0][i*24:(i+1)*24]
tem_base_mean = np.round(np.mean(tem_base,axis = 0),1)
c_0_tem_comfort = [1.4,1.4,1.4,1.4,1.4,1.4,1.2,1.2,1.2,1.2,0.8,0.8,0.8,1.0,1.0,1.0,1.0,1.0,1.2,1.2,1.2,1.4,1.4,1.4]
c_0_tem = np.zeros((24,2,nn_zone))
for ii in range(nn_zone):
    for i in range(24):
        c_0_tem[i,0,ii] = tem_base_mean[i]-c_0_tem_comfort[i]*1.0
        c_0_tem[i,1,ii] = tem_base_mean[i]+c_0_tem_comfort[i]*1.0
J_price,J_upper_tem,J_lower_tem,J_upper_q,J_lower_q = DF_Model_Func.MZ_opt_settings(nn_zone,c_0_tem, SS_P,SS_Y, Train_s,Train_e,Train_s2,Train_e2,predict_period,T_Fre)


J_reco = 0
J_tole = 0.02
J_batch_list = list(range(J_train_samples))
J_batch_size_i = min(J_batch_size,len(J_batch_list))
J_batch_list_i = random.sample(J_batch_list,J_batch_size_i)
time1 = time.time()
for epoch in range(J_epoch):
    for num in range(1):  #(J_batch_num):
        J_optimizer.zero_grad()
        J_batch_data,J_batch_label = [],[]
        for ii in range(S_X_tr.shape[-1]):
            J_batch_data.append(Dis_tr_data[ii][J_batch_list_i,:,:])
            J_batch_label.append(Dis_tr_label[ii][J_batch_list_i,:])
        J_batch_dis_pred,J_cpF = J_model(J_batch_data)
        for ij in range(len(J_batch_dis_pred)):
            J_batch_dis_pred[ij].to(device)
        J_cpF.to(device)

        J_cpA = J_model.model.a.weight
        J_cpB = J_model.model.b
        
        J_obj_list = []
        J_tem = torch.zeros((J_batch_size_i,predict_period+1,nn_zone)).to(device)
        J_tem_re_value = torch.zeros((J_batch_size_i,nn_zone*predict_period)).to(device)
        J_tem_pen = torch.zeros((J_batch_size_i,predict_period+1,nn_zone)).to(device)
        J_q = torch.zeros((J_batch_size_i,predict_period,nn_zone)).to(device)
        J_true_cost = torch.zeros((J_batch_size_i,1)).to(device)
        for c_i_index in range(len(J_batch_list_i)):
            c_period_i = J_batch_list_i[c_i_index]
            c_period_hour = int(predict_period/T_Fre)
            c_index = c_period_i % int(24/c_period_hour)
            c_v_tem,c_v_q,c_v_tem_u,c_p_A,c_p_B,c_p_F,c_prob,c_layer,c_obj1,c_obj2 = DF_Model_Func.MZ_opt_problem(J_price,J_upper_tem,J_lower_tem,J_upper_q,J_lower_q,SS_P, SS_Y, predict_period,T_Fre,c_period_i)
            c_solution = c_layer(J_cpA,J_cpB,J_cpF[c_i_index,:,:])
            c_obj1_i, c_obj2_i, c_obj_i = DF_Model_Func.MZ_opt_objective(T_Fre,SS_P,c_period_i, J_price, c_solution[1],c_solution[2],predict_period)
            J_obj_list.append(c_obj_i)
            J_tem[c_i_index,:,:] = c_solution[0]
            J_q[c_i_index,:,:] = c_solution[1]
            J_tem_pen[c_i_index,:,:] = c_solution[2]
            for ii in range(nn_zone):
                for jj in range(predict_period):
                    J_tem_i = J_tem[c_i_index,jj+1,ii]*(SS_Ymax[ii]-SS_Ymin[ii])+SS_Ymin[ii]
                    J_tem_re_value[c_i_index,ii*predict_period+jj] = (J_tem_i - IC_input_01_all[c_period_i][0,ii*predict_period+jj])/(IC_input_01_all[c_period_i][1,ii*predict_period+jj]-IC_input_01_all[c_period_i][0,ii*predict_period+jj])
            
            J_true_cost[c_i_index,0],_ = J_model.ICNN[c_period_i](J_tem_re_value[c_i_index:c_i_index+1,:])
            print(c_i_index, end='')
        J_loss_dis_i = (1/len(J_batch_dis_pred))*sum(J_lossfn(J_batch_label[i],J_batch_dis_pred[i]) for i in range(len(J_batch_dis_pred)))
        J_loss_obj_i = sum(J_true_cost)/len(J_true_cost)
        J_w_auxi = 1000
        J_loss_i = J_loss_obj_i + J_w_auxi*(J_loss_dis_i)
        J_loss_i.backward()
        J_optimizer.step()
    
    J_loss.append(J_loss_i.item())
    J_loss_obj.append(J_loss_obj_i.item())
    J_loss_dis.append(J_loss_dis_i.item())
    J_models.append([copy.deepcopy(J_model.lstm_list[i].state_dict()) for i in range(S_X_tr.shape[-1])]
                     +[copy.deepcopy(J_model.model.state_dict())])
    if epoch % J_epoch_freq == 0:
        with torch.no_grad():
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}\t{:.6f}'.format(epoch, (num+1) * J_batch_size, J_train_samples, J_loss_obj[epoch], J_loss_dis[epoch]))
    
    if epoch >= 1 and abs(J_loss_obj[epoch]-J_loss_obj[epoch-1]) <= J_tole:
        J_reco += 1
    else:
        J_reco = 0
    if J_reco >= 3:
        break

time2 = time.time()

print('Training time: ',time2-time1)
print('Average Training time: ',(time2-time1)/(epoch+1))

I_model = -1
J_lstm_n = []
for i in range(S_X_tr.shape[-1]):
    J_lstm_n.append(DF_Model_Func.LSTM(P_nn_input,P_nn_hidden,P_nn_layer,P_nn_output).to(device))
    J_lstm_n[i].load_state_dict(J_models[I_model][i])

J_model_n = DF_Model_Func.MZ_model_FNN(M_nn_input, M_nn_hidden, M_nn_output, M_layers).to(device)
J_model_n.load_state_dict(J_models[I_model][-1])

P_train_01 = np.zeros((X_tr.shape[0]-train_period,X_tr.shape[1]))
P_train_acc = np.zeros((S_X_tr.shape[-1],3))
for P_i in range(S_X_tr.shape[-1]):
    P_train_01[:,P_i:P_i+1] = to_np(J_lstm_n[P_i](Dis_tr_data[P_i])).reshape(-1,1)

P_train_huanyuan = SS_X.inverse_transform(P_train_01)
for P_i in range(S_X_tr.shape[-1]):
    P_train_acc[P_i,:] = DF_Model_Func.MZ_LSTM_acc(P_train_huanyuan[:,P_i:P_i+1],X_tr[train_period:,P_i:P_i+1])

P_test_01 = np.zeros((X_te.shape[0]-train_period,X_te.shape[1]))
P_test_acc = np.zeros((S_X_tr.shape[-1],3))
for P_i in range(S_X_tr.shape[-1]):
    P_test_01[:,P_i:P_i+1] = to_np(J_lstm_n[P_i](Dis_te_data[P_i])).reshape(-1,1)

P_test_huanyuan = SS_X.inverse_transform(P_test_01)
for P_i in range(S_X_tr.shape[-1]):
    P_test_acc[P_i,:] = DF_Model_Func.MZ_LSTM_acc(P_test_huanyuan[:,P_i:P_i+1],X_te[train_period:,P_i:P_i+1])

P_trte_acc_sum = np.hstack((P_train_acc,P_test_acc))
P_trte_acc_sum[3:7,2],P_trte_acc_sum[3:7,5],P_trte_acc_sum[1,2],P_trte_acc_sum[1,5] = 1,1,1,1
print(P_trte_acc_sum[0,:])
print(np.mean(P_trte_acc_sum[1:1+22,:],axis = 0))
print(np.mean(P_trte_acc_sum[1+22:,:],axis = 0))     

for i in range(S_X_tr.shape[-1]):
    torch.save(J_models[I_model][i], 'Joint_22zone_LSTM_'+str(i)+'.pt')


M_train_data,M_train_label = DF_Model_Func.MZ_model_data(S_X_tr,S_P_tr,S_Y_tr, train_period, predict_period)
M_test_data,M_test_label = DF_Model_Func.MZ_model_data(S_X_te,S_P_te,S_Y_te, train_period, predict_period)

M_test_pred = to_np(M_model(M_test_data[:-1,:,:])[0])
M_test_pred_1 = M_test_pred.reshape((M_test_pred.shape[0]*M_test_pred.shape[1],M_test_pred.shape[2]), order='F')
M_test_huanyuan = SS_Y.inverse_transform(M_test_pred_1)
M_test_acc = DF_Model_Func.MZ_LSTM_acc(M_test_huanyuan,Y_te[train_period:,:])

M_train_pred = to_np(M_model(M_train_data[:-1,:,:])[0])
M_train_pred_1 = M_train_pred.reshape((M_train_pred.shape[0]*M_train_pred.shape[1],M_train_pred.shape[2]), order='F')
M_train_huanyuan = SS_Y.inverse_transform(M_train_pred_1)
M_train_acc = DF_Model_Func.MZ_LSTM_acc(M_train_huanyuan,Y_tr[train_period:,:])

print('Train RMSE: {},Test RMSE: {}'.format(M_train_acc[0],M_test_acc[0]))

torch.save(J_models[I_model][-1], 'Joint_22zone_Model.pt')

Cr_obj = np.zeros((C_period_train+C_period_test,3))
Cr_q = np.zeros((C_period_train+C_period_test,predict_period,nn_zone))
Cr_tem = np.zeros((C_period_train+C_period_test,predict_period+1,nn_zone))

Seq_input_tr,Seq_input_te = [],[]
for i in range(S_X_tr.shape[1]):
    Seq_input_tr.append(DF_Model_Func.LSTM_sequences(S_X_tr[:,i:i+1], train_period,predict_period))
    Seq_input_te.append(DF_Model_Func.LSTM_sequences(S_X_te[:,i:i+1], train_period,predict_period))

Cr_q_hy = np.zeros((Daily_days+Daily_days_te,Daily_opt*predict_period,nn_zone))
Cr_tem_hy = np.zeros((Daily_days+Daily_days_te,Daily_opt*(predict_period+1),nn_zone))
Opt_time = []
for c_day_index in range(0, Daily_days+Daily_days_te):
    for c_prd_index in range(Daily_opt):
        c_i_index = c_day_index*Daily_opt + c_prd_index
        print(c_i_index)
        c_v_tem,c_v_q,c_v_tem_u,c_p_A,c_p_B,c_p_F,c_prob,c_layer,c_obj1,c_obj2 = DF_Model_Func.MZ_opt_problem(J_price,J_upper_tem,J_lower_tem,J_upper_q,J_lower_q,SS_P, SS_Y, predict_period,T_Fre, c_i_index)
        c_p_A.value = to_np(J_model_n.a.state_dict()['weight'])
        c_p_B.value = to_np(J_model_n.b)

        Seq_output = torch.zeros((predict_period,S_X_tr.shape[-1])).to(device)
        for i in range(S_X_tr.shape[-1]):
            if c_day_index >= Daily_days:
                Seq_input_i = Seq_input_te[i][c_i_index-Daily_days*Daily_opt][0].reshape(1,train_period,1)
            else:
                Seq_input_i = Seq_input_tr[i][c_i_index][0].reshape(1,train_period,1)
            Seq_output[:,i] = J_lstm_n[i](Seq_input_i.to(torch.float32))
        c_p_F.value = to_np(J_model_n.net(Seq_output))
        
        try:
            time3 = time.time()
            c_prob.solve(solver = cp.GUROBI,verbose=False)
            time4 = time.time()
            Opt_time.append(time4-time3)
            Cr_obj[c_i_index,:] = [c_prob.value,c_obj1.value,c_obj2.value]
            Cr_q[c_i_index,:,:] = c_v_q.value
            Cr_tem[c_i_index,:,:] = c_v_tem.value
            Cr_q_hy[c_day_index,predict_period*c_prd_index:predict_period*(c_prd_index+1),:] = SS_P.inverse_transform(c_v_q.value)
            Cr_tem_hy[c_day_index,(predict_period+1)*c_prd_index:(predict_period+1)*(c_prd_index+1),:] = SS_Y.inverse_transform(c_v_tem.value)
        except:
            print('wrong!')
            continue

Cr_obj_trte = [sum(Cr_obj[:C_period_train,0]),sum(Cr_obj[C_period_train:,0])]
Cr_obj_trte1 = [sum(Cr_obj[:C_period_train,1]),sum(Cr_obj[C_period_train:,1])]
Cr_obj_trte2 = [sum(Cr_obj[:C_period_train,2]),sum(Cr_obj[C_period_train:,2])]

print(np.mean(np.array(Opt_time)))
print(np.std(np.array(Opt_time)))

writer = pd.ExcelWriter('Joint_22zone_opt.xlsx')
to_tem_limit = pd.DataFrame(c_0_tem[:,:,0])
to_tem_limit.to_excel(writer,sheet_name='tem_limits',index=False)
for i in range(nn_zone):
    to_Cr_tem = pd.DataFrame(Cr_tem_hy[:,:,i])
    to_Cr_tem.to_excel(writer,sheet_name='tem_trte_'+str(i),index=False)

    to_Cr_q = pd.DataFrame(Cr_q_hy[:,:,i])
    to_Cr_q.to_excel(writer,sheet_name='q_trte_'+str(i),index=False)
writer.close()