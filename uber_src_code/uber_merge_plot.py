import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
from matplotlib.ticker import FuncFormatter

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plot_utilities import LineChart

save_path = "./plots/uber merged plot.png"

series_num = 0
color_list = sns.color_palette()
df = pd.read_csv("./uber csv/uber regret.csv")
fig, ax_list= plt.subplots(2, 3, figsize=(11, 6)) #, gridspec_kw={'height_ratios': [1, 1, 1]}) # , sharex='col'
rho_joint = True # False


def y_fmt(y, pos):
    decades = [1e9, 1e6, 1e3, 1e0, 1e-3, 1e-6, 1e-9 ]
    suffix  = ["G", "M", "k", "" , "m" , "u", "n"  ]
    if y == 0:
        return str(0)
    for i, d in enumerate(decades):
        if np.abs(y) >=d:
            val = y/float(d)
            signf = len(str(val).split(".")[1])
            if signf == 0:
                return '{val:d} {suffix}'.format(val=int(val), suffix=suffix[i])
            else:
                if signf == 1:
                    # print(val, signf)
                    if str(val).split(".")[1] == "0":
                       return '{val:d} {suffix}'.format(val=int(round(val)), suffix=suffix[i]) 
                tx = "{"+"val:.{signf}f".format(signf = signf) +"} {suffix}"
                return tx.format(val=val, suffix=suffix[i])

    return y


### Plot the expected regret of uber and boxplot of the regret of uber
fig, _ = LineChart([df.loc[df["Scenario"]=='Optimal Expectation']["Total $\\rho$"], df.loc[df["Scenario"]=='Optimal Expectation']["Total $\\rho$"]], [df.loc[df["Scenario"]=='Optimal Expectation']["Regret"], df.loc[df["Scenario"]=='Uniform Expectation']["Regret"]]
            , color=[color_list[2], color_list[0]], x_label="Total Incentive $\\rho$", y_label="Regret $\Delta J^c$", legend_loc="upper right", line_style=['-', '--']
            , legend=[] # , legend=["Optimal Expectation", "Uniform Expectation"]
            , ax=ax_list[0, 0]) # , ylim=(0, 200)

x_major_locator = MultipleLocator(1)
ax_list[0, 0].xaxis.set_major_locator(x_major_locator)

### Plot coefficient of uber of different Sources
df2 = pd.read_csv("./uber csv/uber c and rho.csv")
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

### Plot 2-boxplot of the regret of uber
ax = sns.boxplot(ax=ax_list[0, 1], data=df[(df["Total $\\rho$"]<=10) & ( (df["Scenario"]=='Optimal Sample') | (df["Scenario"]=='Uniform Sample') ) ], 
    x="Total $\\rho$", y='Regret', hue='Scenario', palette=[color_list[2], color_list[0]], showfliers = False)
ax.legend().remove()
ax.set_ylabel("Regret $\Delta J^c$")
ax.set_xlabel("Total Incentive $\\rho$")

### Plot 2-boxplot of the forecast errors of uber
df_fore_err = pd.read_csv("./uber csv/uber forecast error.csv")
ax = sns.boxplot(ax=ax_list[1, 2], data=df_fore_err[(df["Total $\\rho$"]<=10) & ( (df_fore_err["Scenario"]=='Optimal Sample') | (df_fore_err["Scenario"]=='Uniform Sample') ) ], 
    x="Total $\\rho$", y='Forecast error', hue='Scenario', palette=[color_list[2], color_list[0]], showfliers = False)
ax.legend().remove()
y_major_locator = MultipleLocator(4)
ax.yaxis.set_major_locator(y_major_locator)
ax.set_ylabel("Forecasting Error $\|\hat{S} - S\|_2$")
ax.set_xlabel("Total Incentive $\\rho$")

### Plot timeseries of uber
S_gt = np.load("./uber numpy data/Sgt{}.npy".format(series_num)).reshape(-1)
# x = np.arange(0, len(S_gt), 1)
#######################  0 == 20140801-10 #######################
dates = pd.date_range('2014/08/01 10:00', periods=len(S_gt), freq='1H')

### plot individual forecasters
if not rho_joint:
    num = 0
    fig2, ax_list2= plt.subplots(1, 2, figsize=(7, 3), sharey='row')
    for rho, marker, op, linew in zip([1, 4], ['o', '^'], [0.3, 0.6], [1, 2]):
        S_hat = np.load("./uber numpy data/S_hat{}_rho{}.npy".format(series_num, rho)).reshape(-1)
        uni_S_hat = np.load("./uber numpy data/uni_S_hat{}_rho{}.npy".format(series_num, rho)).reshape(-1)
        ax_list2[num].plot(dates, S_gt, color='black', linestyle='-.', linewidth=2.5)
        ax_list2[num].plot(dates, S_hat, color=color_list[2], linestyle='-', marker=marker, markersize=4, alpha=op, linewidth=linew)
        ax_list2[num].plot(dates, uni_S_hat, color=color_list[0], linestyle='-', marker=marker, markersize=4, alpha=op, linewidth=linew)
        num += 1
    
    fig2.savefig("./plots/uber rho joint.pdf", bbox_inches='tight')

line_list = [
            Line2D([], [], color='black', linestyle='None', markersize=10, label='$\\rho=1$', alpha=0.3),
            Line2D([], [], color='black', linestyle='None', markersize=10, label='$\\rho=4$', alpha=0.6),
            ]

if rho_joint:
    # for rho, marker, op, linew in zip([3, 4, 5], ['o', '+', '^'], [0.8, 0.6, 0.4], [1, 1.5, 2]):
    num = 0
    for rho, marker, op, linew in zip([1, 4], ['', ''], [0.6, 0.6], [1, 1]):
        S_hat = np.load("./uber numpy data/S_hat{}_rho{}.npy".format(series_num, rho)).reshape(-1)
        uni_S_hat = np.load("./uber numpy data/uni_S_hat{}_rho{}.npy".format(series_num, rho)).reshape(-1)
        ax_list[1, num].plot(dates, S_gt, color='black', linestyle='-.', linewidth=2.5)
        ax_list[1, num].plot(dates, S_hat, color=color_list[2], linestyle='-', marker=marker, markersize=5, alpha=op, linewidth=linew)
        ax_list[1, num].plot(dates, uni_S_hat, color=color_list[0], linestyle='--', marker=marker, markersize=5, alpha=op, linewidth=linew)

        ax_list[1, num].set_ylabel("Uber Pickup Timeseries $S$")
        ax_list[1, num].set_xlabel("Date")
        myFmt = mdates.DateFormatter('%m/%d %H:%M')
        ax_list[1, num].xaxis.set_major_formatter(myFmt)
        ax_list[1, num].xaxis.set_major_locator(mdates.HourLocator(byhour=(10)))
        # ax_list[1, num].legend([line_list[num]], ['$\\rho={}$'.format(3*num+1)], loc="lower right")
        ax_list[1, num].set_title("Total Incentive $\\rho={}$".format(rho))
        ax_list[1, num].yaxis.set_major_formatter(FuncFormatter(y_fmt))
        y_major_locator = MultipleLocator(1000)
        ax_list[1, num].yaxis.set_major_locator(y_major_locator)
        if num == 1:
            ax_list[1, num].set_ylim([-500, 1500])

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
legend_elements[2].set_linestyle("-.")
legend_elements[1].set_linestyle("--")

lgd = fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.) 
    , fancybox=True, shadow=True, ncol=5, fontsize=15)

if rho_joint:
    fig.subplots_adjust(left=0., bottom=0.1, right=1, top=0.89, wspace=0.23, hspace=0.35)
    fig.savefig(save_path, dpi=600, bbox_inches="tight", bbox_extra_artists=(lgd,))
# plt.show()