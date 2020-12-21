import numpy as np
import matplotlib.pyplot as plt
import codecs
import sys

WORK_SPACE = './myLirs/'

markers = ['X', 'o', 'v', '.', '+', '1']
algo = ['LIRS2', 'ML-LIRS', 'LIRS']
colors = ['c', 'm', 'b']

def plot(X, Y, tName, args):
    fig = plt.figure(figsize=(50, 5))
    for i, y in enumerate(Y):
        y = [float(_) for _ in y]
        plt.plot(X, y, color=colors[i], marker=markers[i], label = algo[i], alpha=0.6)
    plt.title(tName)
    plt.xlabel('Epoch')
    plt.ylabel('Miss Ratio (%)')
    plt.legend()
    """
    Set y axis begin at 0
    """
    plt.ylim(bottom=0)
    plt.savefig("../graph/segment_miss/v9/" + tName + "_" + args, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
        
def get_result(path):
    print(path + "..")
    res = []
    t = []
    with open(path, 'r') as f:
        for flt in f.readlines():
            if not flt:
                continue
            t.append( flt )
        
        for tp in t:
            res.append(float(tp))
        return res
    
    
if __name__ == "__main__":
    # if (len(sys.argv) != 3):
    #     raise("Argument Error")
    tName = sys.argv[1]
    cache = sys.argv[2]
    args = sys.argv[3]

    seg_miss_rate_set = []
    # seg_miss_rate_set.append(get_result("../result_set/" + tName + "/lirs_" + tName + "_segment_miss"))
    seg_miss_rate_set.append(get_result("../result_set/" + tName + "/LIRS2_" + cache + "_SEQ_MISS"))
    seg_miss_rate_set.append(get_result("../result_set/" + tName + "/ml_lirs_v9_" + tName + "_" + cache + "_segment_miss"))
    seg_miss_rate_set.append(get_result("../result_set/" + tName + "/LIRS_" + cache + "_SEQ_MISS"))
    # seg_miss_rate_set.append(get_result("../result_set/" + tName + "/opt_" + tName + "_segment_miss"))
    seg_size = [i for i in range(len(seg_miss_rate_set[0]))]
    plot(seg_size, seg_miss_rate_set, tName, cache)


