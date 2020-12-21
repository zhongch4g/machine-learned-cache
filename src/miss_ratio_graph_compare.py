import numpy as np
import matplotlib.pyplot as plt
import codecs
import sys

WORK_SPACE = './myLirs/'

markers = ['X', 'o', 'v', '.', '+', '1']
# algo = ['LRU', 'LIRS', 'ML-LIRS', 'OPT']
# colors = ['r', 'c', 'm', 'b']
algo = ['LRU', 'LIRS', 'ML-LIRS_v6_1', 'ML-LIRS_v9', 'OPT']
colors = ['grey', 'c', '#FF7F0E', '#2077B4', '#D62728', '#343434']

# algo = ['LRU', 'LIRS', 'LIRS2', 'ML-LIRS_v6_1', 'OPT']
# colors = ['grey', 'c', '#FF7F0E', '#2077B4', '#D62728', '#343434']
        
def get_result(path):
    print(path + "..")
    res = []
    t = []
    with open(path, 'r') as f:
        for flt in f.readlines():
            """
            mCacheMaxLimit,
            missRatio, 
            100 - missRatio);
            """
            flt = flt.strip().split(',')
            if not flt:
                continue
            cache_size = int(flt[0].strip('l'))
            t.append( (cache_size, flt[1], flt[2]) )
        t.sort(key=lambda x: x[0])
        
        for tp in t:
            res.append(float(tp[1]))
        return res
    
    
if __name__ == "__main__":
    if (len(sys.argv) != 3):
        raise("Argument Error")
    tName = sys.argv[1]
    args = sys.argv[2]
    
    # Get the trace parameter
    MAX_MEMORY = []
    with codecs.open("../cache_size/" + tName, "r", "UTF8") as inputFile:
    # with codecs.open("/home/zhongchen/myLirs/trace_parameter/" + tName, "r", "UTF8") as inputFile:
        inputFile = inputFile.readlines()
    for line in inputFile:
        if not line == "*\n":
            MAX_MEMORY.append(int(line))

    st = ['1000', '3000', '5000', '7000']

    fig, axs = plt.subplots(1, 4, figsize=(18,6))

    for i, stime in enumerate(st):
        X = MAX_MEMORY
        Y = []
        Y.append(get_result("../result_set/" + tName + "/" + tName + "-LRU"))
        Y.append(get_result("../result_set/" + tName + "/lirs_" + tName))
        # Y.append(get_result("../result_set/" + tName + "/" + tName + "-LIRS2"))
        # Y.append(get_result("../result_set/" + tName + "/ml_lirs_v5_"  + stime + "_"+ tName))
        # Y.append(get_result("../result_set/" + tName + "/ml_lirs_v6_"  + stime + "_"+ tName))
        Y.append(get_result("../result_set/" + tName + "/ml_lirs_v6_1_"  + stime + "_"+ tName))
        Y.append(get_result("../result_set/" + tName + "/ml_lirs_v9_"  + stime + "_"+ tName))
        Y.append(get_result("../result_set/" + tName + "/" + tName + "-OPT"))
        for j, y in enumerate(Y):
            y = [float(_) for _ in y]
            axs[i].plot(X, y, color=colors[j], marker=markers[j], label = algo[j], alpha=0.6)

            axs[i].set_title(tName + "_" + stime)
            axs[i].set_xlabel('Cache Size')
            axs[i].set_ylabel('Miss Rate(%)')
            axs[i].set_ylim(bottom=0)
            axs[i].legend()
    """
    Set y axis begin at 0
    """
    plt.savefig("../graph/miss_ratio/v9/" + tName + args)
    plt.close()




