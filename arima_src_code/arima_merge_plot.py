import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plot_utilities import LineChart

save_path = "./plots/ARIMA merged plot.png"

color_list = sns.color_palette()
series_num = 14
df = pd.read_csv("./arima csv/arima regret.csv")
fig, ax_list= plt.subplots(2, 3, figsize=(11, 6)) #, gridspec_kw={'height_ratios': [1, 1, 1]}) # , sharex='col'

rho_joint = True # False

### Plot the expected regret of ARIMA and boxplot of the regret of ARIMA
fig, _ = LineChart([df.loc[df["Scenario"]=='Optimal Expectation']["Total $\\rho$"], df.loc[df["Scenario"]=='Optimal Expectation']["Total $\\rho$"]], [df.loc[df["Scenario"]=='Optimal Expectation']["Regret"], df.loc[df["Scenario"]=='Uniform Expectation']["Regret"]]
            , color=[color_list[2], color_list[0]], x_label="Total Incentive $\\rho$", y_label="Regret $\Delta J^c$", legend_loc="upper right", line_style=['-', '--']
            , legend=[] # , legend=["Optimal Expectation", "Uniform Expectation"]
            , ax=ax_list[0, 0]) # , ylim=(0, 200)

x_major_locator = MultipleLocator(1)
ax_list[0, 0].xaxis.set_major_locator(x_major_locator)


### Plot coefficient of ARIMA of different forecasters
df2 = pd.read_csv("./arima csv/arima c and rho.csv")
width = 0.25 # the width of the bars: can also be len(x) sequence
# labels = [ str(x) for x in df2.loc[df2["Source"]==0]["Total $\\rho$"].values.tolist()]
labels = df2.loc[df2["Source"]==0]["Total $\\rho$"].values.tolist()
fore1 = df2.loc[df2["Source"]==0]["c"].values.tolist()
fore2 = df2.loc[df2["Source"]==1]["c"].values.tolist()
fore3 = df2.loc[df2["Source"]==2]["c"].values.tolist()
bottom1_2 = [ x + y for x, y in zip(fore1, fore2)]
ax_list[0, 2].bar(labels, fore1, width, color=[color_list[3]]*len(fore1), label="Source 1")
ax_list[0, 2].bar(labels, fore2, width, color=[color_list[6]]*len(fore1), bottom=fore1, label="Source 2")
ax_list[0, 2].bar(labels, fore3, width, color=[color_list[9]]*len(fore1), bottom=bottom1_2, label="Source 3")
ax_list[0, 2].set_ylabel("Mean Coefficient c")
ax_list[0, 2].set_xlabel("Total Incentive $\\rho$")
ax_list[0, 2].legend(["Source 1", "Source 2", "Source 3"], loc="upper right")


### Plot 2-boxplot of the regret of ARIMA
ax = sns.boxplot(ax=ax_list[0, 1], data=df[(df["Total $\\rho$"]<=10) & ( (df["Scenario"]=='Optimal Sample') | (df["Scenario"]=='Uniform Sample') ) ], 
    x="Total $\\rho$", y='Regret', hue='Scenario', palette=[color_list[2], color_list[0]], showfliers = False)
ax.legend().remove()
ax.set_ylabel("Regret $\Delta J^c$")
ax.set_xlabel("Total Incentive $\\rho$")

### Plot 2-boxplot of the forecast errors of ARIMA
df_fore_err = pd.read_csv("./arima csv/arima forecast error.csv")
ax = sns.boxplot(ax=ax_list[1, 2], data=df_fore_err[(df["Total $\\rho$"]<=10) & ( (df_fore_err["Scenario"]=='Optimal Sample') | (df_fore_err["Scenario"]=='Uniform Sample') ) ], 
    x="Total $\\rho$", y='Forecast error', hue='Scenario', palette=[color_list[2], color_list[0]], showfliers = False)
ax.legend().remove()
ax.set_ylabel("Forecasting Error $\|\hat{S} - S\|_2$")
ax.set_xlabel("Total Incentive $\\rho$")

### Plot timeseries of ARIMA
S_gt = np.load("./arima numpy data/Sgt{}.npy".format(series_num))
x = np.arange(0, len(S_gt), 1)
ax_list[1, 0].plot(x, S_gt, color='black', linestyle='-.', linewidth=2.5)


### plot individual forecasters
if not rho_joint:
    num = 0
    fig2, ax_list2= plt.subplots(1, 2, figsize=(7, 3), sharey='row')
    for rho, marker, op, linew in zip([1, 4], ['o', '^'], [0.3, 0.6], [1, 2]):
        S_hat = np.load("./arima numpy data/S_hat{}_rho{}.npy".format(series_num, rho))
        uni_S_hat = np.load("./arima numpy data/uni_S_hat{}_rho{}.npy".format(series_num, rho))
        ax_list2[num].plot(x, S_gt, color='black', linestyle='-.', linewidth=2.5)
        ax_list2[num].plot(x, S_hat, color=color_list[2], linestyle='-', marker=marker, markersize=4, alpha=op, linewidth=linew)
        ax_list2[num].plot(x, uni_S_hat, color=color_list[0], linestyle='-', marker=marker, markersize=4, alpha=op, linewidth=linew)
        num += 1
    
    fig2.savefig("./plots/ARIMA rho joint.pdf", bbox_inches='tight')

# line_list = [
#             Line2D([], [], color='black', marker='o', linestyle='None', markersize=10, label='$\\rho=1$', alpha=0.3),
#             Line2D([], [], color='black', marker='^', linestyle='None', markersize=10, label='$\\rho=4$', alpha=0.6),
#             ]

if rho_joint:
    num = 0
    # for rho, marker, op, linew in zip([3, 4, 5], ['o', '+', '^'], [0.8, 0.6, 0.4], [1, 1.5, 2]):
    for rho, marker, op, linew in zip([1, 4], ['', ''], [0.6, 0.6], [1, 1]):
        S_hat = np.load("./arima numpy data/S_hat{}_rho{}.npy".format(series_num, rho))
        uni_S_hat = np.load("./arima numpy data/uni_S_hat{}_rho{}.npy".format(series_num, rho))
        ax_list[1, num].plot(x, S_gt, color='black', linestyle='-.', linewidth=2.5)
        ax_list[1, num].plot(x, S_hat, color=color_list[2], linestyle='-', marker=marker, markersize=4, alpha=op, linewidth=linew)
        ax_list[1, num].plot(x, uni_S_hat, color=color_list[0], linestyle='--', marker=marker, markersize=4, alpha=op, linewidth=linew)   
       
        ax_list[1, num].set_ylabel("ARIMA Timeseries $S$")
        ax_list[1, num].set_xlabel("Time steps $t$")
        # ax_list[1, num].set_ylim((1.2, 1.7))
        x_major_locator = MultipleLocator(1)
        ax_list[1, num].xaxis.set_major_locator(x_major_locator)
        # ax_list[1, num].legend([line_list[num]], ['$\\rho={}$'.format(3*num+1)], loc="lower right")
        ax_list[1, num].set_title("Total Incentive $\\rho={}$".format(rho))

        num += 1

legend_elements = [
                    Line2D([0], [0], color=color_list[2], label="ACS (Ours)"),
                    Line2D([0], [0], color=color_list[0], label="Uniform"),
                    Line2D([0], [0], color="black"  , label="Ground True"),
                    # Line2D([], [], color='black', marker='o', linestyle='None', markersize=10, label='$\\rho=1$', alpha=0.3),
                    # Line2D([], [], color='black', marker='^', linestyle='None', markersize=10, label='$\\rho=4$', alpha=0.6),
                #    Line2D([], [], color='black', marker='^', linestyle='None', markersize=10, label='$\\rho=5$'),
                #    Line2D([0], [0], color=color_list[3], label="Source 1", lw = 7),
                #    Line2D([0], [0], color=color_list[4], label="Source 2", lw = 7),
                #    Line2D([0], [0], color=color_list[5], label="Source 3", lw = 7),
                   ]

legend_elements[1].set_linestyle("--")
legend_elements[2].set_linestyle("-.")

lgd = fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.) 
    , fancybox=True, shadow=True, ncol=5, fontsize=15)

if rho_joint:
    fig.subplots_adjust(left=0., bottom=0.1, right=1, top=0.89, wspace=0.2, hspace=0.35)
    fig.savefig(save_path, dpi=600, bbox_inches="tight", bbox_extra_artists=(lgd,))

# plt.show()