import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yaml
import pickle
import torch
import numpy as np

import sys
import os
from tsai.all import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch.manual_seed(0)
np.random.seed(0)

def TrainTSForecaster(ts, w, h, split_size, arch, name, epoch=60, lr=1e-3):
    X, y = SlidingWindow(window_size, horizon=horizon)(ts)
    splits = TimeSplitter(split_size, show_plot=False)(y)
    # print(X.shape, y.shape) # (2922, 4, window) (2922, 4, horizon=5)
    batch_tfms = TSStandardize()
    fcst = TSForecaster(X, y, splits=splits, path='models', batch_tfms=batch_tfms, 
        bs=512, arch=arch, metrics=mae)
    fcst.fit_one_cycle(epoch, lr)
    raw_preds, target, preds = fcst.get_X_preds(X[splits[0]], y[splits[0]])
    # print(raw_preds.shape, target.shape) # [1459, 4, 5], [1459, 4, 5]
    raw_preds, target = raw_preds.reshape(-1, 20), target.reshape(-1, 20)
    # print(raw_preds.shape, raw_predsT.shape) # [1459, 20], [20, 1459]

    error = (raw_preds - target).cpu().detach().numpy()
    cov = np.cov(error, rowvar=False)
    mean = np.mean(error, axis=0)
    with open('./uber numpy data/Cov{}_h{}_w{}.npy'.format(name, h, w), 'wb') as f:
        np.save(f, cov)
    print(cov.shape, np.linalg.norm(cov), mean)
    # print(type(raw_preds), type(target)) # fastai.torch_core.TensorBase, torch.Tensor
    
    fcst.export("forecaster{}_h{}_w{}.pkl".format(name, h, w))

    return

if __name__ == "__main__":
    window_size_list = [5, 7, 10]
    arch_list = [GRUPlus, InceptionTimePlus, TSTPlus]
    arch_name_list = ["GRUPlus", "InceptionTimePlus", "TSTPlus"]

    yaml_file = open("parameters.yml")
    parsed_yaml_file = yaml.load(yaml_file, Loader=yaml.FullLoader)["uber_para"]
    horizon = parsed_yaml_file['T']
    data_direct = parsed_yaml_file['data_direct']
    scalerfile = parsed_yaml_file['scalerfile']

    ### load csv to pd
    train_df = pd.read_csv(data_direct + "uber training data.csv")
    train_ts = train_df.to_numpy()
    test_df = pd.read_csv(data_direct + "uber testing data.csv")
    test_ts = test_df.to_numpy()

    ### training data shape: [# samples x # variables x sequence length]
    ### sample_n x 4 x windowsize
    train_ts = train_ts[:, -1].reshape(-1, 4).astype(np.float64)
    test_ts = test_ts[:, -1].reshape(-1, 4).astype(np.float64)
    with open('./uber numpy data/training_ts.npy', 'wb') as f1:
        np.save(f1, train_ts)

    with open('./uber numpy data/testing_ts.npy', 'wb') as f2:
        np.save(f2, test_ts)

    ts = np.concatenate((train_ts, test_ts), axis=0)
    split_size = test_ts.shape[0]
    # print(train_ts.shape, test_ts.shape, ts.shape) # (2928, 4) (1463, 4) (4391, 4)
    # print(ts[0, :5])

    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    min_max_scaler.fit(train_ts)
    train_ts_norm = min_max_scaler.transform(train_ts)
    pickle.dump(min_max_scaler, open(scalerfile, 'wb'))

    for window_size, arch, name in zip(window_size_list, arch_list, arch_name_list):
        TrainTSForecaster(train_ts_norm, window_size, horizon, split_size, arch, name)
