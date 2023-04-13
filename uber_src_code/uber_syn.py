import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler
import pickle
import torch
from tsai.all import *

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src_class import src
from ACS import ACS
from uber_system_dynamics import CtrlTask, CtrlTaskIndentity

### src 1 lowest feature: past time series
### src 2 medium feature: past time series 
### src 3 biggest feature: past time series

yaml_file = open("parameters.yml")
parsed_yaml_file = yaml.load(yaml_file, Loader=yaml.FullLoader)["uber_para"]
n = parsed_yaml_file['n']
m = parsed_yaml_file['m']
p = parsed_yaml_file['p']
T = parsed_yaml_file['T']
horizon = T
scalerfile = parsed_yaml_file['scalerfile']

np.random.seed(42)

n_src = 3
arch_name_list = ["TSTPlus", "InceptionTimePlus", "GRUPlus"]
window_size_list = [10, 7, 5]
shift_list = [0, 3, 5]
horizon = parsed_yaml_file['T']

min_max_scaler = pickle.load(open(scalerfile, 'rb'))
test_ts = np.load('./uber numpy data/testing_ts.npy') # (1463, 4)
test_ts_norm = min_max_scaler.transform(test_ts)

def Inverse_transform(scaled_data, data_min, data_max): # shape: 24x1, 1x4, 1x4

    return scaled_data * (data_max[0] - data_min[0]) + data_min[0]

def FullControlVector(vec): # vec: (sample_n, p, T) ->  (sample_n, 1, p*T)
    sample_n = vec.shape[0]
    S_full = np.zeros((sample_n, 1, p*T))
    vec_ = vec.reshape(-1, 1, p*T)
    flag = 0
    for tt in range(T):
        S_full[:, :, flag*p:flag*p + p] = vec_[:, :, flag:p*T:T]
        flag += 1
        
    return S_full


Psi, _ = CtrlTaskIndentity(np.zeros((1, p*T)), T=T)

#### window size is diffrent so target y is shifted for different forecasters
# for jj in range(100):
#     for ii in range(n_src):
#         past, futu = SlidingWindow(window_size_list[ii], horizon=horizon)(test_ts_norm)
#         print(futu.shape) # 1449x4x5, 1454x4x5, 1452x4x5
#         print(futu[jj+shift_list[ii], 0, :])
#     input()


src_list = []
past_list = []
futu_list = []
for ii in range(n_src):
    src_list.append(src(T, 1, 2*(ii+1), (1*ii+1.5), 0.3*(ii+1), "src{}".format(ii)))
    src_list[-1].forecaster = load_learner("models/forecaster{}_h{}_w{}.pkl".format(arch_name_list[ii], horizon, window_size_list[ii]), cpu=False)
    src_list[-1].covar = np.load("uber numpy data/Cov{}_h{}_w{}.npy".format(arch_name_list[ii], horizon, window_size_list[ii]))
    past, futu = SlidingWindow(window_size_list[ii], horizon=horizon)(test_ts_norm)
    past_list.append(past)
    futu_list.append(futu)

df = pd.DataFrame({'Total $\\rho$' : [], 'Scenario': [], 'Regret': []})
df2 = pd.DataFrame({'Total $\\rho$' : [], 'Source': [], 'c': [], 'rho': []})
df3 = pd.DataFrame({'Total $\\rho$' : [], 'Scenario': [], 'Forecast error': []})
rho_range = np.arange(0, 5.5, 0.5, dtype="object")

for rho_total in rho_range:
    ### expected regret
    c, rho, regret, uni_regret = ACS(Psi, src_list, rho_total)
    for i_src in range(n_src):
        df2.loc[len(df2.index)] = [str(rho_total), str(i_src), np.mean(c[i_src, :, 0]), rho[i_src]]

    df.loc[len(df.index)] = [str(rho_total), 'Optimal Expectation', regret]
    df.loc[len(df.index)] = [str(rho_total), 'Uniform Expectation', uni_regret]

    saveFlag = True if int(rho_total) == rho_total else False
    
    ### empirical regret
    ### full future control vector s1t1, s2t1, s3t1, s4t1, s1t2
    S_gt = futu_list[2][shift_list[2]:, :, :]
    S_gt = FullControlVector(S_gt)
    sample_n = S_gt.shape[0]
    S_hat = np.zeros((sample_n, 1, p*T))
    uni_S_hat = np.zeros((sample_n, 1, p*T))
    c_uni = np.ones((n_src, p*T, 1)) / n_src
    rho_uni = np.ones((n_src)) / n_src * rho_total

    for i_src, src_ in enumerate(src_list):
        ii = shift_list[i_src] # sample num
        past, futu = past_list[i_src][ii:, :, :], futu_list[i_src][ii:, :, :]
        # print(futu.shape) # 1449x4x5
        fore, target, _ = src_.forecaster.get_X_preds(past, futu)
        fore, target = fore.cpu().detach().numpy(), target.cpu().detach().numpy()
        fore = FullControlVector(fore)
        target = FullControlVector(target)
        laplace_var = src_.deltaS * (1 + np.exp(-src_.beta * (rho[i_src] - src_.gamma) ) ) / src_.alpha
        laplace_noise = np.random.laplace(0, laplace_var, size=fore.shape)
        S_hat_i = fore + laplace_noise
        S_hat += c[i_src, :, :].T * S_hat_i

        uni_laplace_var = src_.deltaS * (1 + np.exp(-src_.beta * (rho_uni[i_src] - src_.gamma) ) ) / src_.alpha
        uni_laplace_noise = np.random.laplace(0, uni_laplace_var, size=fore.shape)
        uni_S_hat_i = fore + uni_laplace_noise
        uni_S_hat += c_uni[i_src, :, :].T * uni_S_hat_i

    if saveFlag:

        with open('./uber numpy data/S_hat{}_rho{}.npy'.format(0, int(rho_total)), 'wb') as f:
            S_hat_scaled = Inverse_transform(S_hat[0:48, :, 0], min_max_scaler.data_min_, min_max_scaler.data_max_)
            np.save(f, S_hat_scaled)

        with open('./uber numpy data/uni_S_hat{}_rho{}.npy'.format(0, int(rho_total)), 'wb') as f:
            uni_S_hat_scaled = Inverse_transform(uni_S_hat[0:48, :, 0], min_max_scaler.data_min_, min_max_scaler.data_max_)
            np.save(f, uni_S_hat_scaled)

        with open('./uber numpy data/Sgt{}.npy'.format(0), 'wb') as f:
            S_gt_scaled = Inverse_transform(S_gt[0:48, :, 0], min_max_scaler.data_min_, min_max_scaler.data_max_)
            np.save(f, S_gt_scaled)
        
        saveFlag = False

    deltaJ_array = np.diag( (S_hat - S_gt).reshape(-1, 20) @ Psi @ (S_hat - S_gt).reshape(-1, 20).T )
    uni_deltaJ_array = np.diag( (uni_S_hat - S_gt).reshape(-1, 20) @ Psi @ (uni_S_hat - S_gt).reshape(-1, 20).T )

    for i_sample, (deltaJ, uni_deltaJ) in enumerate(zip(deltaJ_array, uni_deltaJ_array)):
        df.loc[len(df.index)] = [str(rho_total), 'Optimal Sample', deltaJ]
        df.loc[len(df.index)] = [str(rho_total), 'Uniform Sample', uni_deltaJ]
        df3.loc[len(df3.index)] = [str(rho_total), 'Optimal Sample', np.linalg.norm( (S_hat - S_gt)[i_sample].reshape(-1, 20) , ord=2)]
        df3.loc[len(df3.index)] = [str(rho_total), 'Uniform Sample', np.linalg.norm( (uni_S_hat - S_gt)[i_sample].reshape(-1, 20), ord=2)]

df.to_csv("./uber csv/uber regret.csv", index=False)
df2.to_csv("./uber csv/uber c and rho.csv", index=False)
df3.to_csv("./uber csv/uber forecast error.csv", index=False)
