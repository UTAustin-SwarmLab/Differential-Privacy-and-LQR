# Differential-Privacy-and-LQR
ACC 2023 submission. [link]()

Multiple differential Private forecasters for LQR.

## Prerequisites

python version: 3.9.12
|package|version|
|:-----:|------:|
|numpy| 1.23.1|
|pandas| 1.4.4|
|PyYAML| 6.0|
|matplotlib| 3.5.3|
|tsai| 0.3.1|
|torch|1.11.0|
|scikit-learn| 1.1.2|
|seaborn| 0.12.0|
|cvxpy| 1.2.1|

For more details, see ```environment.yaml```.

## Getting Started
Pls run all the source file in the root directory :)

### Experiments
* Go to the root directory of the repo.

#### ARIMA
* To run experiments:
```arima_syn.py```

* To plot:
```arima_merge_plot.py```

#### Uber
* Pre-process uber data from the "uber data" directory. Please download the data [here](https://www.kaggle.com/datasets/fivethirtyeight/uber-pickups-in-new-york-city), and execute:
```uber_preprocessing.py```

* To train timeseries forecast models:
```uber_train_forecaster.py```

* To run experiments:
```uber_syn.py```

* To plot:
```uber_merge_plot.py```

### Other functions
* Src code of input-driven LQR controllers and ARIMA time series:
```system_dynamics.py```

* Dimension of action space, state space, and other parameters:
```parameters.yml```

* Functions to plot:
```plot_utilities.py```

* Alternative Convex Search algorithm:
```ACS.py```

* Class for timeseries sources:
```src_class.py```

## Contact

Po-han Li - pohanli@utexas.edu
