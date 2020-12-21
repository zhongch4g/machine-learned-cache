import re
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dashes=[(2,2), (4,1), (2,0), (2,0), (1,1)]
algo = ['1', '2', '3', '4', '4+', 'L']
# colors = ['tomato', 'seagreen', 'c', 'lightsalmon', '#2077B4', 'y']
colors = ['r', 'g', 'c', 'k']
hatch = ['o', 'xxx', '-',  'O','\\', '/']

def plot(cache, row, Y1, Y2, trace):
    fig, ax = plt.subplots(figsize=(16, 3.6))
    # fig,ax = plt.subplots()
    # print(Y1)
    # k = [i for i in range(row)]
    k = np.arange(row)
  
    width = 0.2
    ax.bar(k, Y1[0], color=colors[0], label='OUT_IN', width=width)
    ax.bar(k, Y1[1], bottom=Y1[0], color=colors[1], label='OUT_OUT', width=width)
    ax.bar(k, Y1[2], bottom=Y1[0] + Y1[1], color=colors[2], label='IN_IN', width=width)
    ax.bar(k, Y1[3], bottom=Y1[0] + Y1[1] + Y1[2], color=colors[3], label='IN_OUT', width=width)

    ax.bar(k+width, Y2[0], color=colors[0], label='OUT_IN', width=width)
    ax.bar(k+width, Y2[1], bottom=Y2[0], color=colors[1], label='OUT_OUT', width=width)
    ax.bar(k+width, Y2[2], bottom=Y2[0] + Y2[1], color=colors[2], label='IN_IN', width=width)
    ax.bar(k+width, Y2[3], bottom=Y2[0] + Y2[1] + Y2[2], color=colors[3], label='IN_OUT', width=width)

    # for tick in ax.get_xticklabels():
    #     tick.set_rotation(45)
    ax.set_xticks(k)
    # ax.set_xticklabels(k)
    ax.tick_params(axis="y", direction="inout", pad=-25)
    # ax.autoscale(tight=True) 
    ax.set_title(trace + "_" + str(cache), position=(0.5, 1.1))
    ax.set_xlabel('Epoch')
    # ax.set_ylabel('blocks/ref_time(%)')
    # ax.legend(loc="upper left")
    ax.legend(["OUT_IN", "OUT_OUT", "IN_IN", "IN_OUT"], loc="upper right", fontsize=18,
edgecolor='k',facecolor='k', framealpha=0, mode="expand", ncol=4, bbox_to_anchor=(0.05, 1.1, 0.9, .102))

    fig.savefig("../graph/breakdown/" + trace + "_" + str(cache) + "_tt.pdf", bbox_inches='tight', pad_inches=0)
    plt.close()

def get_par_data(path):
    par = []
    with open(path, "r") as f:
        for line in f.readlines():
            number = line.strip().split('\n')[0]
            if not number:
                continue
            # par.append((int(number) * 4096)/pow(2, 30))
            par.append(int(number))
        return par

print(sys.argv)
par_traces = [sys.argv[1]]
# args = "_".join(sys.argv[2:])
args = [int(i) for i in sys.argv[2:]]
print(args)
if (not par_traces):
    par_traces = ['zigzag']

pattr_set = []
for trace in par_traces:
    print("Get par from ../cache_size/" + trace)
    cache_size = get_par_data("../cache_size/" + trace)
    print("Get par finished..")
    # cache_size = [400, 700, 1000, 1300, 1800]
    if (len(args) > 0):
        cache_size = args
    # cache_size = [400, 1000, 1400]
    # cache_size = [400]
    for cache in cache_size:
        header_list = ['total', 'out_out', 'out_in', 'in_out', 'in_in']
        opt = pd.read_csv('../result_set/' + trace + '/' + 'ml_lirs_v9_' + trace + '_' + str(cache) + '_breakdown_opt', names=header_list) 
        # print(file)
        opt['out_out'] = opt['out_out'].astype(float)
        opt['out_in'] = opt['out_in'].astype(float)
        opt['in_out'] = opt['in_out'].astype(float)
        opt['in_in'] = opt['in_in'].astype(float)
        opt = opt[['in_out', 'in_in', 'out_out', 'out_in']]
        # print(opt)
        opt_percentage = opt.div(opt.sum(axis=1), axis=0)
        opt_percentage = opt_percentage * 100
        opt_percentage = opt_percentage.fillna(0)
        # print(opt_percentage)

        header_list = ['total', 'out_out', 'out_in', 'in_out', 'in_in']
        ml_lirs = pd.read_csv('../result_set/' + trace + '/' + 'ml_lirs_v9_' + trace + '_' + str(cache) + '_breakdown_ml_lirs', names=header_list) 
        ml_lirs['out_out'] = ml_lirs['out_out'].astype(float)
        ml_lirs['out_in'] = ml_lirs['out_in'].astype(float)
        ml_lirs['in_out'] = ml_lirs['in_out'].astype(float)
        ml_lirs['in_in'] = ml_lirs['in_in'].astype(float)
        ml_lirs = ml_lirs[['in_out', 'in_in', 'out_out', 'out_in']]
        # print(ml_lirs)
        ml_lirs_percentage = ml_lirs.div(ml_lirs.sum(axis=1), axis=0)
        ml_lirs_percentage = ml_lirs_percentage * 100
        ml_lirs_percentage = ml_lirs_percentage.fillna(0)
        # print(ml_lirs_percentage)

        # opt = [opt_percentage['in_out'].values, opt_percentage['in_in'].values, opt_percentage['out_out'].values, opt_percentage['out_in'].values]
        opt = [opt_percentage['out_in'].values, opt_percentage['out_out'].values, opt_percentage['in_in'].values, opt_percentage['in_out'].values]
        # ml_lirs = [ml_lirs_percentage['in_out'].values, ml_lirs_percentage['in_in'].values, ml_lirs_percentage['out_out'].values, ml_lirs_percentage['out_in'].values]
        ml_lirs = [ml_lirs_percentage['out_in'].values, ml_lirs_percentage['out_out'].values, ml_lirs_percentage['in_in'].values, ml_lirs_percentage['in_out'].values]
        # print(opt, opt_percentage.shape[0])
        print("Start plot..")
        plot(cache, opt_percentage.shape[0], opt, ml_lirs, trace)

    
print('end....')
