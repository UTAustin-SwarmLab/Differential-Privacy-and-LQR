import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plot_utilities import LineChart

save_path = "./plots/ARIMA incentive vs regret.pdf"
save_path2 = "./plots/ARIMA incentive vs c.pdf"
save_path3 = "./plots/ARIMA incentive vs c two box.pdf"
save_path4 = "./plots/ARIMA incentive vs forecast error.pdf"
save_path5 = "./plots/ARIMA incentive vs forecast.pdf"


color_list = sns.color_palette()

df = pd.read_csv("./arima csv/arima regret.csv")

fig, ax = LineChart([df.loc[df["Scenario"]=='Optimal Expectation']["Total $\\rho$"], df.loc[df["Scenario"]=='Optimal Expectation']["Total $\\rho$"]], [df.loc[df["Scenario"]=='Optimal Expectation']["Regret"], df.loc[df["Scenario"]=='Uniform Expectation']["Regret"]]
            , color=[color_list[2], color_list[0]], title="Incentive v.s. Regret", x_label="Total Incentive $\\rho$", y_label="Regret $\Delta J^c$", legend_loc="upper right", line_style=['-', '--'], legend=["Optimal Expectation", "Uniform Expectation"]
            , save_path="./plots/ARIMA incentive vs regret no box.pdf") # , ylim=(0, 200)

# ax.scatter(x=df.loc[df["Scenario"]=='Optimal Sample']["Total $\\rho$"], y=df.loc[df["Scenario"]=='Optimal Sample']["Regret"], s=3, c='tomato')
# ax.scatter(x=df.loc[df["Scenario"]=='Uniform Sample']["Total $\\rho$"], y=df.loc[df["Scenario"]=='Uniform Sample']["Regret"], s=3, c='springgreen')

for rho in df.loc[df["Scenario"]=='Optimal Expectation']["Total $\\rho$"]:
    # if rho - int(rho) == 0:
    # print(rho, df[(df["Scenario"]=='Optimal Sample') & (df["Total $\\rho$"]==rho) ]["Regret"])
    box = ax.boxplot(x=[df[(df["Scenario"]=='Optimal Sample') & (df["Total $\\rho$"]==rho) ]["Regret"]], 
                positions=[rho], showfliers=False, meanline=False, showmeans=False)

# x_major_locator = MultipleLocator(1)
# ax.xaxis.set_major_locator(x_major_locator)
plt.xticks(range(0,6), range(0,6))

fig.savefig(save_path, dpi=1000)

fig.clf()
# fig, ax = plt.subplots()
# markers = [".", "x", "+"]
# colors = ["r", "g", "b"]
# for i_src in range(3):
#     ax.scatter(x = df2.loc[df2["Source"]==i_src]["rho"], y = df2.loc[df2["Source"]==i_src]["c"], c=colors[i_src], marker = markers[i_src])

# ax.set_xlabel("Coefficient c")
# ax.set_ylabel("Incentive $\\rho$")
# ax.legend(["Forecaster 1", "Forecaster 2", "Forecaster 3"], loc="lower right")
# fig.savefig(save_path2, dpi=1000)

df2 = pd.read_csv("./arima csv/arima c and rho.csv")
fig, ax = plt.subplots()
width = 0.15 # the width of the bars: can also be len(x) sequence
# labels = [ str(x) for x in df2.loc[df2["Source"]==0]["Total $\\rho$"].values.tolist()]
labels = df2.loc[df2["Source"]==0]["Total $\\rho$"].values.tolist()
fore1 = df2.loc[df2["Source"]==0]["c"].values.tolist()
fore2 = df2.loc[df2["Source"]==1]["c"].values.tolist()
fore3 = df2.loc[df2["Source"]==2]["c"].values.tolist()
bottom1_2 = [ x + y for x, y in zip(fore1, fore2)]
ax.bar(labels, fore1, width, color=[color_list[3]]*len(fore1), label="Forecaster 1")
ax.bar(labels, fore2, width, color=[color_list[4]]*len(fore1), bottom=fore1, label="Forecaster 2")
ax.bar(labels, fore3, width, color=[color_list[5]]*len(fore1), bottom=bottom1_2, label="Forecaster 3")

ax.set_ylabel("Coefficient c")
ax.set_xlabel("Total Incentive $\\rho$")
ax.legend(["Forecaster 1", "Forecaster 2", "Forecaster 3"], loc="upper right")
fig.savefig(save_path2, dpi=1000)

fig.clf()

# x=[df[(df["Scenario"]=='Optimal Sample') & (df["Total $\\rho$"]==rho) ]["Regret"]]
ax = sns.boxplot(data=df[(df["Total $\\rho$"]<=10) & ( (df["Scenario"]=='Optimal Sample') | (df["Scenario"]=='Uniform Sample') ) ], 
    x="Total $\\rho$", y='Regret', hue='Scenario', palette=[color_list[2], color_list[0]], showfliers = False)
ax.set_ylabel("Regret $\Delta J^c$")
ax.set_xlabel("Total Incentive $\\rho$")

fig.savefig(save_path3, dpi=1000)

fig.clf()

df_fore_err = pd.read_csv("./arima csv/arima forecast error.csv")

ax = sns.boxplot(data=df_fore_err[(df["Total $\\rho$"]<=10) & ( (df_fore_err["Scenario"]=='Optimal Sample') | (df_fore_err["Scenario"]=='Uniform Sample') ) ], 
    x="Total $\\rho$", y='Forecast error', hue='Scenario', palette=[color_list[2], color_list[0]], showfliers = False)
ax.set_ylabel("Forecast Error $\|S_hat - S\|_2$")
ax.set_xlabel("Total Incentive $\\rho$")

fig.savefig(save_path4, dpi=1000)

fig.clf()
fig, ax = plt.subplots()
S_gt = np.load("./arima numpy data/Sgt95.npy")
x = np.arange(0, len(S_gt), 1)
ax.plot(x, S_gt, color='black', linestyle='-.', linewidth=2.5)

for rho, marker, op, linew in zip([0, 3, 5], ['o', '+', '^'], [0.8, 0.6, 0.4], [1, 1.5, 2]):
    S_hat = np.load("./arima numpy data/S_hat95_rho{}.npy".format(rho))
    uni_S_hat = np.load("./arima numpy data/uni_S_hat95_rho{}.npy".format(rho))
    ax.plot(x, S_hat, color=color_list[2], linestyle='-', marker=marker, markersize=3, alpha=op, linewidth=linew)
    ax.plot(x, uni_S_hat, color=color_list[0], linestyle='--', marker=marker, markersize=3, alpha=op, linewidth=linew)


ax.set_ylabel("Timeseries $S$")
ax.legend(['Ground True', 'Optimal $\\rho=0$', 'Uniform $\\rho=0$', 'Optimal $\\rho=3$', 
    'Uniform $\\rho=3$', 'Optimal $\\rho=5$', 'Uniform $\\rho=5$'], loc='lower right', ncol=3)
ax.set_title("Timeseries and Forecasts")
# ax.set_ylim((-15, 10))
fig.savefig(save_path5, dpi=1000)
