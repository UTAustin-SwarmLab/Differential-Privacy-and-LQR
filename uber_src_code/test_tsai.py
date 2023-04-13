from tsai.all import *
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"

window_size = 60
horizon = 5

# for window_size in window_size_list:
### training
ts = get_forecasting_time_series("Sunspots").values
# print(ts.shape) # 3235, 1
X, y = SlidingWindow(window_size, horizon=horizon)(ts)
splits = TimeSplitter(235)(y) # 235 is the size of testing set
# print(len(splits), splits[1])
# input()
batch_tfms = TSStandardize()
### loss is always MSE, MAE is the metrics
# fcst = TSForecaster(X, y, splits=splits, path='models', batch_tfms=batch_tfms, bs=512, arch=TSTPlus, metrics=mae, cbs=ShowGraph())
fcst = TSForecaster(X, y, splits=splits, path='models', batch_tfms=batch_tfms, bs=512, arch=TSTPlus, metrics=mse)
fcst.fit_one_cycle(50, 1e-3) # 50 is epoch
fcst.export("fcst.pkl")

# [# samples x # variables x sequence length]
print(X[splits[1]].shape) # 235, 1, 60


### inference
from tsai.inference import load_learner

fcst = load_learner("models/fcst.pkl", cpu=False)
raw_preds, target, preds = fcst.get_X_preds(X[splits[0]], y[splits[0]])
print(raw_preds.shape) # 2936, 5 = target shape = raw_preds shape


### test
test_pred, target, preds = fcst.get_X_preds(X[splits[1]], y[splits[1]])
print(test_pred.shape) # 235, 5

# input()
# print(len(preds)) # len is 2396conda rename -n old_name -d new_name 
