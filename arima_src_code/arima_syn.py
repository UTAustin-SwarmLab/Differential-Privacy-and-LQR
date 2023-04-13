import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src_class import src
from ACS import ACS
from system_dynamics import ARIMA_gen, CtrlTask, CtrlTaskIndentity

### src 1 lowest feature: past time series
### src 2 medium feature: past time series 
### src 3 biggest feature: past time series

yaml_file = open("parameters.yml")
parsed_yaml_file = yaml.load(yaml_file, Loader=yaml.FullLoader)["parameters"]
T = parsed_yaml_file['T']
sample_n = parsed_yaml_file['sample_n']

np.random.seed(42)

n_src = 3
feature_gap = 3

S_scaled, mu, __, ___, ____ = ARIMA_gen(sample_n=500, T=2*T)
argmean = np.argmax(mu)


S_min, S_max = np.min(S_scaled, axis=1), np.max(S_scaled, axis=1)
S_min, S_max = np.repeat(S_min, 2*T).reshape(S_scaled.shape), np.repeat(S_max, 2*T).reshape(S_scaled.shape)
S = ( S_scaled - S_min ) / ( S_max - S_min )
S_min, S_max = S_min[:, T:], S_max[:, T:]
# print(S_min.shape, S_max.shape) # (500, 20) (500, 20)

S_past = S[:, 0:T]
S_futu = S[:, T:]

# max_min = np.max(S_futu, axis=1)
# print(max_min.shape, max_min)
# arg = np.argmax(max_min)
# print(arg, max_min[arg])
# input()

# for sample in np.random.randint(low=0, high=sample_n, size=200):
#     print(sample)
#     plot(x1, S_scaled[])

Psi, _ = CtrlTaskIndentity(S_futu[0, :], T=T)

# Psi, _ = CtrlTask(S_futu[0, :], T=T, random_seed=42)

src_list = []
for ii in range(n_src):
    #########
    feature = S_past[:, ii*feature_gap:]
    #########
    regr = LinearRegression().fit(feature, S_futu)
    src_list.append(src(T, 1, 4*(ii+1), (1*ii+1.5), 0.3*(ii+1), "src{}".format(ii)))
    src_list[-1].forecaster = regr
    src_list[-1].MeasureErrorCovar(feature, S_futu)

df = pd.DataFrame({'Total $\\rho$' : [], 'Scenario': [], 'Regret': []})
df2 = pd.DataFrame({'Total $\\rho$' : [], 'Source': [], 'c': [], 'rho': []})
df3 = pd.DataFrame({'Total $\\rho$' : [], 'Scenario': [], 'Forecast error': []})
rho_range = np.arange(0, 5.5, 0.5, dtype="object")

ii_sample = np.random.randint(low=0, high=sample_n, size=200)
ii_sample[0] = 14

for rho_total in rho_range:
    ### expected regret
    c, rho, regret, uni_regret = ACS(Psi, src_list, rho_total)
    for i_src in range(n_src):
        df2.loc[len(df2.index)] = [str(rho_total), str(i_src), np.mean(c[i_src, :, 0]), rho[i_src]]

    df.loc[len(df.index)] = [str(rho_total), 'Optimal Expectation', regret]
    df.loc[len(df.index)] = [str(rho_total), 'Uniform Expectation', uni_regret]

    saveFlag = True if int(rho_total) == rho_total else False
    ### empirical regret
    for ii in ii_sample:
        n_src = len(src_list)
        S_gt = S_futu[ii, :].reshape(1, -1)
        S_hat = np.zeros((1, T))
        uni_S_hat = np.zeros((1, T))
        c_uni = np.ones((n_src, T, 1)) / n_src
        rho_uni = np.ones((n_src)) / n_src * rho_total

        for i_src, src_ in enumerate(src_list):
            #########
            fore = src_.forecaster.predict(S_past[ii, i_src*feature_gap:].reshape(1, -1))
            #########

            laplace_var = src_.deltaS * (1 + np.exp(-src_.beta * (rho[i_src] - src_.gamma) ) ) / src_.alpha
            laplace_noise = np.random.laplace(0, laplace_var, size=fore.shape)
            S_hat_i = fore + laplace_noise
            S_hat += c[i_src, :, :].T * S_hat_i

            uni_laplace_var = src_.deltaS * (1 + np.exp(-src_.beta * (rho_uni[i_src] - src_.gamma) ) ) / src_.alpha
            uni_laplace_noise = np.random.laplace(0, uni_laplace_var, size=fore.shape)
            uni_S_hat_i = fore + uni_laplace_noise
            uni_S_hat += c_uni[i_src, :, :].T * uni_S_hat_i

        S_hat_scaled = S_hat * ( S_max[ii, :] - S_min[ii, :] ) + S_min[ii, :]
        uni_S_hat_scaled = uni_S_hat * ( S_max[ii, :] - S_min[ii, :] ) + S_min[ii, :]
        S_gt_scaled = S_gt * ( S_max[ii, :] - S_min[ii, :] ) + S_min[ii, :]
            
        if saveFlag:
            with open('./arima numpy data/S_hat{}_rho{}.npy'.format(ii, int(rho_total)), 'wb') as f:
                np.save(f, S_hat_scaled.reshape(-1))

            with open('./arima numpy data/uni_S_hat{}_rho{}.npy'.format(ii, int(rho_total)), 'wb') as f:
                np.save(f, uni_S_hat_scaled.reshape(-1))

            with open('./arima numpy data/Sgt{}.npy'.format(ii), 'wb') as f:
                np.save(f, S_gt_scaled.reshape(-1))

            saveFlag = False

        deltaJ = ( (S_hat - S_gt) @ Psi @ (S_hat - S_gt).T ).item(0)
        uni_deltaJ = ( (uni_S_hat - S_gt) @ Psi @ (uni_S_hat - S_gt).T ).item(0)
        df.loc[len(df.index)] = [str(rho_total), 'Optimal Sample', deltaJ]
        df.loc[len(df.index)] = [str(rho_total), 'Uniform Sample', uni_deltaJ]
        df3.loc[len(df3.index)] = [str(rho_total), 'Optimal Sample', np.linalg.norm(S_hat_scaled - S_gt_scaled, ord=2)]
        df3.loc[len(df3.index)] = [str(rho_total), 'Uniform Sample', np.linalg.norm(uni_S_hat_scaled - S_gt_scaled, ord=2)]


df.to_csv("./arima csv/arima regret.csv", index=False)
df2.to_csv("./arima csv/arima c and rho.csv", index=False)
df3.to_csv("./arima csv/arima forecast error.csv", index=False)
