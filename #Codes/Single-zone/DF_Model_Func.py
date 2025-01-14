#%%
import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device("cuda:3")
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def data_input(T_Fre,Train_s,Train_e,Train_s2,Train_e2,train_period,N_zone):
    T_len = 8760*T_Fre
    
    if N_zone == 1:
        if T_Fre == 4:
            data_in = np.loadtxt("2zone_15min.csv",delimiter=",",skiprows=1,usecols=range(1,12))
        P0 = data_in[:,10:11]/1000
        T_in0 = data_in[:,8:9]
        T_o0 = data_in[:,0:1]
        T_occ0 = data_in[:,2:3]/1000
        T_rad0 = data_in[:,4:5]/1000

    if N_zone == 22 or N_zone == 90:
        if T_Fre == 4:
            data_in = np.loadtxt(str(N_zone)+'zone_15min.csv',delimiter=",",skiprows=1,usecols=range(1,5*N_zone+2))
    
        P0,T_in0 = np.zeros((T_len,N_zone)),np.zeros((T_len,N_zone))
        T_rad0 = data_in[:,1+N_zone*1:1+N_zone*2]/1000
        for ii in range(N_zone):
            P0[:,ii] = data_in[:,3+N_zone*2+3*ii]/1000
            T_in0[:,ii] = data_in[:,1+N_zone*2+3*ii]
        T_o0 = data_in[:,0].reshape(-1,1)
        T_occ0 = data_in[:,1:1+N_zone*1].copy()/1000

    X_tr = np.hstack((T_o0[(Train_s-train_period):Train_e,0:1],T_rad0[(Train_s-train_period):Train_e,:],T_occ0[(Train_s-train_period):Train_e,:]))
    P_tr = P0[(Train_s-train_period):Train_e,:]
    Y_tr = T_in0[(Train_s-train_period):Train_e,:]
    
    X_te = np.hstack((T_o0[(Train_s2-train_period):Train_e2,0:1],T_rad0[(Train_s2-train_period):Train_e2,:],T_occ0[(Train_s2-train_period):Train_e2,:]))
    P_te = P0[(Train_s2-train_period):Train_e2,:]
    Y_te = T_in0[(Train_s2-train_period):Train_e2,:]

    SS_X = MinMaxScaler().fit(X_tr)
    SS_P = MinMaxScaler().fit(P_tr)
    SS_Y = MinMaxScaler().fit(Y_tr)

    S_X_tr = SS_X.transform(X_tr)
    S_P_tr = SS_P.transform(P_tr)
    S_Y_tr = SS_Y.transform(Y_tr)
    S_X_te = SS_X.transform(X_te)
    S_P_te = SS_P.transform(P_te)
    S_Y_te = SS_Y.transform(Y_te)

    return SS_X,SS_P,SS_Y,S_X_tr,S_P_tr,S_Y_tr,S_X_te,S_P_te,S_Y_te,X_tr,X_te,Y_tr,Y_te

def LSTM_sequences(input_data, train_period,predict_period):
    inout_seq = []
    L = len(input_data)
    for i in range(0,L-train_period,predict_period):
        train_seq = torch.tensor(input_data[i:i+train_period,:]).to(device)
        train_label = torch.tensor(input_data[i+train_period:i+train_period+predict_period,:]).to(device)
        inout_seq.append((train_seq, train_label))
    return inout_seq

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:,-1,:])
        return predictions

def LSTM_acc(Y_te_pre,Y_te):
    T_len = Y_te.shape[0]
    Err_tr = np.abs(Y_te_pre[:,0] - Y_te[:,0])
    Err_tr1 = math.sqrt(sum(Err_tr[i]**2/(T_len) for i in range(T_len)))  #RMSE
    Err_tr2 = sum(Err_tr[i] for i in range(T_len))/(T_len)  #MAE
    Err_tr3 = r2_score(Y_te, Y_te_pre)
    ERR2 = [Err_tr1,Err_tr2,Err_tr3]
    return ERR2

def model_data(S_X_tr,S_P_tr,S_Y_tr, train_period, predict_period):
    N_sample = int((S_X_tr.shape[0]-train_period)/predict_period)
    model_data = np.zeros((predict_period, N_sample, S_X_tr.shape[-1]+S_P_tr.shape[-1]+S_Y_tr.shape[-1]))
    model_label = np.zeros((predict_period,N_sample))
    for i in range(N_sample):
        i_start = train_period+i*predict_period
        model_data[:,i,0] = S_Y_tr[i_start:i_start+predict_period,0]
        model_data[:,i,1:2] = S_P_tr[i_start:i_start+predict_period,:]
        model_data[:,i,2:] = S_X_tr[i_start:i_start+predict_period,:]
        model_label[:,i] = S_Y_tr[i_start:i_start+predict_period,0]
    model_data = torch.tensor(model_data).to(device)
    model_label = torch.tensor(model_label).to(device)
    return model_data,model_label

class model_FNN(nn.Module):
    def __init__(self,input_num,hidden_units,output_num,layer_num):  #input:时长97; output:40
        super(model_FNN, self).__init__()
        self.input_num,self.hidden_units,self.output_num = input_num,hidden_units,output_num
        self.layer_num = layer_num
        self.a = nn.Parameter(torch.tensor(0.92), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(-0.3), requires_grad=True)
        layers = []
        layers.append(nn.Linear(self.input_num, self.hidden_units))
        layers.append(nn.ReLU())
        for _ in range(self.layer_num-1):
            layers.append(nn.Linear(self.hidden_units, self.hidden_units))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_units, self.output_num))
        self.net = nn.Sequential(*layers)
    
    def forward(self,u):
        u_y = torch.zeros(u.shape[0]+1,u.shape[1], dtype=torch.float64).to(device)
        u_y[0,:] = copy.deepcopy(u[0,:,0])
        for j in range(u_y.shape[0]-1):
            input_new = torch.zeros(u.shape[1],u.shape[2]).to(device)
            if j == 0:
                input_new[:,:] = u[j,:,:]
            else:
                input_new[:,1:] = u[j,:,1:]
                input_new[:,0] = u_y[j,:]
            u_out = self.a*input_new[:,0:1] + self.b*input_new[:,1:2] + self.net(input_new[:,2:])
            u_y[j+1:j+2,:] = u_out.T
        return u_y

class CvxModel(nn.Module):
    def __init__(self, n_feature=2, n_hidden=4, n_output=1):
        super(CvxModel, self).__init__()
        self.input_layer = nn.Linear(n_feature, n_hidden, bias=False)
        self.hidden_layer1 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.hidden_layer2 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.output_layer = nn.Linear(n_hidden, n_output, bias=False)
        self.passthrough_layer1 = nn.Linear(n_feature, n_hidden)
        self.passthrough_layer2 = nn.Linear(n_feature, n_hidden)
        self.passthrough_output = nn.Linear(n_feature, n_output)

    def forward(self, x):
        zx1 = nn.ReLU()(self.input_layer(x))
        pass1 = self.passthrough_layer1(x)
        pass2 = self.passthrough_layer2(x)
        pass_out = self.passthrough_output(x)
        zx2 = nn.ReLU()(self.hidden_layer1(zx1) + pass1)
        zx3 = nn.ReLU()(self.hidden_layer2(zx2) + pass2)        
        zx_out = self.output_layer(zx3) + pass_out
        return self.input_layer(x),zx1,pass1,zx_out
    
class weightConstraint():
    def __init__(self):
        pass

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(0.0, 100.0)
            module.weight.data = w

class FNNModel(nn.Module):
    def __init__(self, n_feature=2, n_hidden=4, n_output=1):
        super(FNNModel, self).__init__()
        self.input_layer = nn.Linear(n_feature, n_hidden)
        self.hidden_layer1 = nn.Linear(n_hidden, n_hidden)
        self.hidden_layer2 = nn.Linear(n_hidden, n_hidden)
        self.output_layer = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        zx1 = nn.ReLU()(self.input_layer(x))
        zx2 = nn.ReLU()(self.hidden_layer1(zx1))
        zx3 = nn.ReLU()(self.hidden_layer2(zx2))        
        zx_out = self.output_layer(zx3)
        return zx_out

class opt_joint_map_all(nn.Module):
    def __init__(self, P_models,M_model,IC_map_all):
        super(opt_joint_map_all, self).__init__()
        self.lstm1 = P_models[0]
        self.lstm2 = P_models[1]
        self.lstm3 = P_models[2]
        self.model = M_model
        self.ICNN = IC_map_all

    def forward(self, Dis_out,  Dis_rad,  Dis_occ):
        D1_pred = self.lstm1(Dis_out)
        D2_pred = self.lstm2(Dis_rad)
        D3_pred = self.lstm3(Dis_occ)
        c_p_F = torch.zeros(D1_pred.shape[0],D1_pred.shape[1]).to(device)
        for i in range(D1_pred.shape[1]):
            input = torch.cat((D1_pred[:,i:i+1],D2_pred[:,i:i+1],D3_pred[:,i:i+1]),dim = 1).to(device)
            c_p_F[:,i:i+1] = self.model.net(input)
        return [D1_pred,D2_pred,D3_pred],c_p_F

def opt_settings(n_zone,c_0_tem, SS_P,SS_Y, Train_s,Train_e,Train_s2,Train_e2,predict_period,T_Fre):
    c_0_price = np.loadtxt("data_2023.csv",delimiter=",",skiprows=1,usecols=range(1, 16))[::int(12/T_Fre),:][Train_s:Train_e2,9]
    c_period_train,c_period_test = int((Train_e-Train_s)/predict_period),int((Train_e2-Train_s2)/predict_period)
    c_period_hour = int(predict_period/T_Fre)
    c_time = predict_period + 1

    c_01_tem_reshape = [SS_Y.transform(c_0_tem[:,0].reshape(-1,1)).reshape(int(24/c_period_hour),c_period_hour),SS_Y.transform(c_0_tem[:,1].reshape(-1,1)).reshape(int(24/c_period_hour),c_period_hour)]
    c_upper_tem = np.zeros(((int(24/c_period_hour),c_time)))
    c_lower_tem = np.zeros((int(24/c_period_hour),c_time))
    for i in range(c_upper_tem.shape[0]):
        c_upper_tem[i,:-1] = np.repeat(c_01_tem_reshape[1][i,:],T_Fre)
        c_lower_tem[i,:-1] = np.repeat(c_01_tem_reshape[0][i,:],T_Fre)
    for i in range(c_upper_tem.shape[0]-1):
        c_upper_tem[i,-1],c_lower_tem[i,-1] = c_upper_tem[i+1,0],c_lower_tem[i+1,0]
    c_upper_tem[-1,-1],c_lower_tem[-1,-1] = c_upper_tem[0,0],c_lower_tem[0,0]

    c_price = np.zeros((c_period_train+c_period_test,predict_period))
    for i in range(c_period_train+c_period_test):
        for j in range(predict_period):
            c_price[i,j] = c_0_price[i*predict_period+j]/1000
    
    c_upper_p = 35*np.ones((predict_period,n_zone))
    c_lower_p = 0*np.ones((predict_period,n_zone))
    c_PI = 3.60
    c_upper_q = SS_P.transform(c_PI*c_upper_p)
    c_lower_q = SS_P.transform(c_PI*c_lower_p)
    return c_price,c_upper_tem,c_lower_tem,c_upper_q,c_lower_q

def opt_problem(c_price,c_upper_tem,c_lower_tem,c_upper_q,c_lower_q,SS_P, SS_Y, predict_period,T_Fre,c_period_i):
    c_period_hour = int(predict_period/T_Fre)
    c_time = predict_period + 1
    SS_Pmax,SS_Pmin,SS_Ymax,SS_Ymin = SS_P.data_max_[0],SS_P.data_min_[0],SS_Y.data_max_[0],SS_Y.data_min_[0]

    c_tem_cost_u = 0.4
    c_v_tem,c_v_q = cp.Variable(c_time),cp.Variable(predict_period)
    c_v_tem_u = cp.Variable(c_time, pos=True)

    c_p_A,c_p_B,c_p_F = cp.Parameter(),cp.Parameter(),cp.Parameter(predict_period)
    
    c_PI = 3.60
    c_obj1 = sum(c_price[c_period_i,t]*(1/c_PI)*(1/T_Fre)*((SS_Pmax-SS_Pmin)*c_v_q[t]+SS_Pmin) for t in range(predict_period))
    c_obj2 = sum(c_tem_cost_u*(1/T_Fre)*(c_v_tem_u[t]**2) for t in range(c_time))
    c_obj = c_obj1 + c_obj2

    c_index = c_period_i % int(24/c_period_hour)
    cons_tem1 = [c_v_tem[t] <= c_upper_tem[c_index,t] + c_v_tem_u[t]/(SS_Ymax-SS_Ymin) for t in range(c_time)]
    cons_tem2 = [c_v_tem[t] >= c_lower_tem[c_index,t] - c_v_tem_u[t]/(SS_Ymax-SS_Ymin) for t in range(c_time)]
    cons_q = [c_v_q <= c_upper_q[:,0], c_v_q >= c_lower_q[:,0]]
    cons_model = [c_v_tem[t+1] == c_p_A*c_v_tem[t] + c_p_B*c_v_q[t] + c_p_F[t] for t in range(predict_period)]
    cons_ini1 = [c_v_tem[0] == 0.5*(c_upper_tem[c_index,0]+c_lower_tem[c_index,0])]
    cons = cons_tem1 + cons_tem2 + cons_q + cons_model + cons_ini1

    c_prob = cp.Problem(cp.Minimize(c_obj), cons)
    c_layer = CvxpyLayer(c_prob, parameters=[c_p_A,c_p_B,c_p_F], variables=[c_v_tem,c_v_q,c_v_tem_u])
    return c_v_tem,c_v_q,c_v_tem_u,c_p_A,c_p_B,c_p_F,c_prob,c_layer,c_obj1,c_obj2

def opt_objective(T_Fre,SS_P,c_period_i, c_price, c_v_q,c_v_tem_u,predict_period):
    c_time = predict_period + 1
    c_tem_cost_u = 0.4
    SS_Pmax,SS_Pmin = SS_P.data_max_[0],SS_P.data_min_[0]
    c_PI = 3.60
    c_obj1 = sum(c_price[c_period_i,t]*(1/c_PI)*(1/T_Fre)*((SS_Pmax-SS_Pmin)*c_v_q[t]+SS_Pmin) for t in range(predict_period))
    c_obj2 = sum(c_tem_cost_u*(1/T_Fre)*(c_v_tem_u[t]**2) for t in range(c_time))
    c_obj = c_obj1 + c_obj2
    return c_obj1, c_obj2, c_obj


#%%
class MZ_CvxModel(nn.Module):
    def __init__(self, n_zone,n_feature=2, n_hidden=4, n_output=1, predict_period = 32):
        super(MZ_CvxModel, self).__init__()
        self.predict_period = predict_period
        self.w = nn.Parameter((1/n_zone)*torch.ones((n_zone, 1)), requires_grad=True)
        self.input_layer = nn.Linear(n_feature, n_hidden, bias=False)
        self.hidden_layer1 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.hidden_layer2 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.output_layer = nn.Linear(n_hidden, n_output, bias=False)
        self.passthrough_layer1 = nn.Linear(n_feature, n_hidden)
        self.passthrough_layer2 = nn.Linear(n_feature, n_hidden)
        self.passthrough_output = nn.Linear(n_feature, n_output)

    def forward(self, x0):
        nn_zone = int(x0.shape[1]/self.predict_period)
        x = sum(self.w[j,0] * x0[:,self.predict_period*j:self.predict_period*(j+1)] for j in range(nn_zone))
        zx1 = nn.ReLU()(self.input_layer(x))
        pass1 = self.passthrough_layer1(x)
        pass2 = self.passthrough_layer2(x)
        pass_out = self.passthrough_output(x)
        zx2 = nn.ReLU()(self.hidden_layer1(zx1) + pass1)
        zx3 = nn.ReLU()(self.hidden_layer2(zx2) + pass2)
        zx_out = self.output_layer(zx3) + pass_out

        relu = nn.ReLU()
        comp = [sum(relu(-1*self.w[i,0])**2 for i in range(nn_zone)), (1-sum(self.w))**2]
        return zx_out, comp

class MZ_FNNModel(nn.Module):
    def __init__(self, n_zone, n_feature=2, n_hidden=4, n_output=1, predict_period = 32):
        super(MZ_FNNModel, self).__init__()
        self.zone = n_zone
        self.predict_period = predict_period
        self.w = nn.Parameter((1/self.zone)*torch.ones((self.zone, 1)), requires_grad=True)
        self.input_layer = nn.Linear(n_feature, n_hidden)
        self.hidden_layer1 = nn.Linear(n_hidden, n_hidden)
        self.hidden_layer2 = nn.Linear(n_hidden, n_hidden)
        self.output_layer = nn.Linear(n_hidden, n_output)

    def forward(self, x0):
        x = sum(self.w[j,0] * x0[:,self.predict_period*j:self.predict_period*(j+1)] for j in range(self.zone))
        
        zx1 = nn.ReLU()(self.input_layer(x))
        zx2 = nn.ReLU()(self.hidden_layer1(zx1))
        zx3 = nn.ReLU()(self.hidden_layer2(zx2))        
        zx_out = self.output_layer(zx3)
        
        relu = nn.ReLU()
        comp = [sum(relu(-1*self.w[i,0])**2 for i in range(self.zone)), (1-sum(self.w))**2]
        return zx_out,comp


def MZ_LSTM_acc(Y_te_pre,Y_te):
    T_len = Y_te.shape[0]
    Err_tr = np.abs(Y_te_pre[:,:] - Y_te[:,:])
    Err_tr1 = [math.sqrt(sum(Err_tr[i,j]**2/(T_len) for i in range(T_len))) for j in range(Y_te.shape[1])] #RMSE
    Err_tr2 = [sum(Err_tr[i,j] for i in range(T_len))/(T_len) for j in range(Y_te.shape[1])]  #MAE
    Err_tr3 = [r2_score(Y_te[:,j], Y_te_pre[:,j])   for j in range(Y_te.shape[1])]
    ERR2 = [sum(Err_tr1)/len(Err_tr1),sum(Err_tr2)/len(Err_tr2),sum(Err_tr3)/len(Err_tr3)]
    return ERR2

def MZ_model_data(S_X_tr,S_P_tr,S_Y_tr, train_period, predict_period):
    N_sample = int((S_X_tr.shape[0]-train_period)/predict_period)
    model_data = np.zeros((predict_period, N_sample, S_X_tr.shape[-1]+S_P_tr.shape[-1]+S_Y_tr.shape[-1]))
    model_label = np.zeros((predict_period,N_sample,S_Y_tr.shape[-1]))
    for i in range(N_sample):
        i_start = train_period+i*predict_period
        model_data[:,i,:S_Y_tr.shape[-1]] = S_Y_tr[i_start:i_start+predict_period,:]
        model_data[:,i,S_Y_tr.shape[-1]:S_Y_tr.shape[-1]+S_P_tr.shape[-1]] = S_P_tr[i_start:i_start+predict_period,:]
        model_data[:,i,S_Y_tr.shape[-1]+S_P_tr.shape[-1]:] = S_X_tr[i_start:i_start+predict_period,:]
        model_label[:,i,:] = S_Y_tr[i_start:i_start+predict_period,:]
    model_data = torch.tensor(model_data).to(device)
    model_label = torch.tensor(model_label).to(device)
    return model_data,model_label

class MZ_model_FNN(nn.Module):
    def __init__(self,input_num,hidden_units,output_num,layer_num):
        super(MZ_model_FNN, self).__init__()
        self.input_num,self.hidden_units,self.output_num = input_num,hidden_units,output_num
        self.layer_num = layer_num

        self.a = nn.Linear(self.output_num, self.output_num, bias=False)#nn.Parameter(0.92*torch.ones((self.output_num, self.output_num)), requires_grad=True)
        nn.init.constant_(self.a.weight, 0.003)  #0.03  #Please check here for a good initial value: (0.003 for 90-zone and 0.03 for 22-zone)
        self.b = nn.Parameter(-0.01*torch.ones((self.output_num, 1)), requires_grad=True) #-0.3  #Also, -0.01 for 90-zone and -0.3 for 22-zone
        layers = []
        layers.append(nn.Linear(self.input_num, self.hidden_units))
        layers.append(nn.ReLU())
        for _ in range(self.layer_num-1):
            layers.append(nn.Linear(self.hidden_units, self.hidden_units))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_units, self.output_num))
        self.net = nn.Sequential(*layers)
    
    def forward(self,u):
        u_y = torch.zeros(u.shape[0]+1,u.shape[1],self.output_num, dtype=torch.float64).to(device)
        u_y[0,:,:] = copy.deepcopy(u[0,:,:self.output_num])
        for j in range(u_y.shape[0]-1):
            input_new = torch.zeros(u.shape[1],u.shape[2]).to(device)
            if j == 0:
                input_new[:,:] = u[j,:,:]
            else:
                input_new[:,self.output_num:] = u[j,:,self.output_num:]
                input_new[:,:self.output_num] = u_y[j,:,:]
            for i2 in range(u_y.shape[2]):
                u_y[j+1,:,i2] = self.a(input_new[:,:self.output_num])[:,i2] + self.b[i2,0]*input_new[:,self.output_num+i2] + self.net(input_new[:,self.output_num+self.output_num:])[:,i2]
        relu = nn.ReLU()
        comp_pen = relu(self.b)**2
        return u_y,comp_pen

def MZ_opt_settings(n_zone,c_0_tem, SS_P,SS_Y, Train_s,Train_e,Train_s2,Train_e2,predict_period,T_Fre):
    c_0_price = np.loadtxt("data_2023.csv",delimiter=",",skiprows=1,usecols=range(1, 16))[::int(12/T_Fre),:][Train_s:Train_e2,9]
    c_period_train,c_period_test = int((Train_e-Train_s)/predict_period),int((Train_e2-Train_s2)/predict_period)
    c_period_hour = int(predict_period/T_Fre)
    c_time = predict_period + 1

    c_01_tem_reshape = [SS_Y.transform(c_0_tem[:,0,:]).reshape(int(24/c_period_hour),c_period_hour,c_0_tem.shape[-1]),SS_Y.transform(c_0_tem[:,1,:]).reshape(int(24/c_period_hour),c_period_hour,c_0_tem.shape[-1])]
    c_upper_tem = np.zeros(((int(24/c_period_hour),c_time,c_0_tem.shape[-1])))
    c_lower_tem = np.zeros((int(24/c_period_hour),c_time,c_0_tem.shape[-1]))
    for i in range(c_upper_tem.shape[0]):
        for j in range(c_upper_tem.shape[1]-1):
            c_upper_tem[i,j,:] = c_01_tem_reshape[1][i,j//T_Fre,:]
            c_lower_tem[i,j,:] = c_01_tem_reshape[0][i,j//T_Fre,:]
    for i in range(c_upper_tem.shape[0]-1):
        c_upper_tem[i,-1,:],c_lower_tem[i,-1,:] = c_upper_tem[i+1,0,:],c_lower_tem[i+1,0,:]
    c_upper_tem[-1,-1,:],c_lower_tem[-1,-1,:] = c_upper_tem[0,0,:],c_lower_tem[0,0,:]

    c_price = np.zeros((c_period_train+c_period_test,predict_period))
    for i in range(c_period_train+c_period_test):
        for j in range(predict_period):
            c_price[i,j] = c_0_price[i*predict_period+j]/1000
    
    c_upper_p = 35*np.ones((predict_period,n_zone))
    if n_zone == 90:
        c_upper_p[:,[8,17,26,35,44,53,62,71,80,89]] = 0
    c_lower_p = 0*np.ones((predict_period,n_zone))
    c_PI = 3.60
    c_upper_q = SS_P.transform(c_PI*c_upper_p)
    c_lower_q = SS_P.transform(c_PI*c_lower_p)
    return c_price,c_upper_tem,c_lower_tem,c_upper_q,c_lower_q

def MZ_opt_problem(c_price,c_upper_tem,c_lower_tem,c_upper_q,c_lower_q,SS_P, SS_Y, predict_period,T_Fre,c_period_i):
    nn_zone = c_upper_q.shape[-1]
    c_period_hour = int(predict_period/T_Fre)
    c_time = predict_period + 1
    SS_Pmax,SS_Pmin,SS_Ymax,SS_Ymin = [SS_P.data_max_[i] for i in range(nn_zone)],[SS_P.data_min_[i] for i in range(nn_zone)],[SS_Y.data_max_[i] for i in range(nn_zone)],[SS_Y.data_min_[i] for i in range(nn_zone)]

    c_tem_cost_u = 0.4
    c_v_tem,c_v_q = cp.Variable((c_time,nn_zone)),cp.Variable((predict_period,nn_zone))
    c_v_tem_u = cp.Variable((c_time,nn_zone), pos=True)

    c_p_A,c_p_B,c_p_F = cp.Parameter((nn_zone,nn_zone)),cp.Parameter((nn_zone,1)),cp.Parameter((predict_period,nn_zone))
    
    c_PI = 3.60
    c_obj1 = (1/c_PI)*(1/T_Fre)*sum((SS_Pmax[ii]-SS_Pmin[ii])*c_price[c_period_i:c_period_i+1,:]@c_v_q[:,ii:ii+1]+SS_Pmin[ii] for ii in range(nn_zone))
    c_obj2 = c_tem_cost_u*(1/T_Fre)*cp.sum(c_v_tem_u**2)
    c_obj = c_obj1 + c_obj2

    c_index = c_period_i % int(24/c_period_hour)
    cons_tem1 = [c_v_tem[:,ii] <= c_upper_tem[c_index,:,ii] + c_v_tem_u[:,ii]/(SS_Ymax[ii]-SS_Ymin[ii]) for ii in range(nn_zone)]
    cons_tem2 = [c_v_tem[:,ii] >= c_lower_tem[c_index,:,ii] - c_v_tem_u[:,ii]/(SS_Ymax[ii]-SS_Ymin[ii]) for ii in range(nn_zone)]
    cons_q = [c_v_q <= c_upper_q, c_v_q >= c_lower_q]
    cons_model = [c_v_tem[t+1:t+2,:] == c_v_tem[t:t+1,:]@c_p_A.T + cp.multiply(c_v_q[t:t+1,:],c_p_B[:,0:1].T) + c_p_F[t:t+1,:] for t in range(predict_period)]
    cons_ini1 = [c_v_tem[0,:] == 0.5*(c_upper_tem[c_index,0,:]+c_lower_tem[c_index,0,:])]
    cons = cons_tem1 + cons_tem2 + cons_q + cons_model + cons_ini1

    c_prob = cp.Problem(cp.Minimize(c_obj), cons)
    c_layer = CvxpyLayer(c_prob, parameters=[c_p_A,c_p_B,c_p_F], variables=[c_v_tem,c_v_q,c_v_tem_u])
    return c_v_tem,c_v_q,c_v_tem_u,c_p_A,c_p_B,c_p_F,c_prob,c_layer,c_obj1,c_obj2

def MZ_opt_objective(T_Fre,SS_P,c_period_i, c_price, c_v_q,c_v_tem_u,predict_period):
    nn_zone = c_v_q.shape[-1]
    c_time = predict_period + 1
    c_tem_cost_u = 0.4
    SS_Pmax,SS_Pmin = [SS_P.data_max_[i] for i in range(nn_zone)],[SS_P.data_min_[i] for i in range(nn_zone)]
    c_PI = 3.60
    c_obj1 = sum(c_price[c_period_i,t]*(1/c_PI)*(1/T_Fre)*sum((SS_Pmax[ii]-SS_Pmin[ii])*c_v_q[t,ii]+SS_Pmin[ii] for ii in range(nn_zone)) for t in range(predict_period))
    c_obj2 = sum(c_tem_cost_u*(1/T_Fre)*sum((c_v_tem_u[t,ii]**2) for ii in range(nn_zone)) for t in range(c_time))
    c_obj = c_obj1 + c_obj2
    return c_obj1, c_obj2, c_obj

