import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src_class import src
from ACS import ACS
from system_dynamics import ARIMA_gen, CtrlTask, CtrlTaskIndentity, SVD

### src 1 lowest var feature: past time series
### src 2 medium var feature: past time series 
### src 3 biggest var feature: past time series

yaml_file = open("parameters.yml")
parsed_yaml_file = yaml.load(yaml_file, Loader=yaml.FullLoader)["parameters"]
# T = parsed_yaml_file['T']
T_list = [5, 10, 100]
sample_n = parsed_yaml_file['sample_n']

np.random.seed(42)
for T in T_list:
    save_path = "./plots/test_s_codesign{}.pdf".format(T)

    S, _, __, ___, ____ = ARIMA_gen(sample_n=500, T=2*T)
    S_past = S[:, 0:T]
    S_futu = S[:, T:]
    Psi, _ = CtrlTaskIndentity(S_futu[0, :], T=T)

    gauss = np.random.normal(loc=0.0, scale=1.0, size=(T, 2000))

    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    min_max_scaler.fit(gauss)
    gauss2 = min_max_scaler.transform(gauss)
    print(np.max(gauss2), np.min(gauss2))

    gauss1D = np.mean(gauss, axis=0)
    ctrl_cost = np.diag( gauss.T @ Psi @ gauss )
    # ctrl_cost = ( gauss.T @ Psi @ gauss ).item()
    # print(ctrl_cost, gauss1D.shape)

    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    axs[0].hist(gauss1D, bins=20)
    axs[1].hist(ctrl_cost, bins=20)
    fig.savefig(save_path, dpi=1000)

