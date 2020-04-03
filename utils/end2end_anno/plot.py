import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from data import load_data, load_data_word
import numpy as np
import tensorflow as tf
from numpy import linalg as LA
import time
# ----------------------------------------------------------------------------
maxlen0 = 60
maxlen1 = 3
load_gradient = True
# ----------------------------------------------------------------------------
maxwlen = 10
cmaxlen0 = maxlen0  * (maxwlen
                     + 2         # start/end of word symbol
                     + 1)        # whitespace between tokens
cmaxlen1 = maxlen1 * (maxwlen
                     + 2         # start/end of word symbol
                     + 1)        # whitespace between tokens
# ---------------------------------------------------------------------------

import random
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

path = os.path.abspath(__file__)
save_path = os.path.dirname(path).replace('utils/end2end_anno',
                                          'data/end2end')

def plot(data='test', anno_path=save_path): 
   
    fig = plt.figure(figsize=(10, 20))
    gs = gridspec.GridSpec(10, 10, wspace=0.1, hspace=1)
    st = fig.suptitle("char&word model", fontsize="x-large") 
    file_name = '%s_gradient.pdf'%data
    gradient_path = '/nfs_shares/wzw0022_home/%s_gradient_norm.npz'%data
    npz = np.load(gradient_path)
    wnorms = npz['wnorms']
    cnorms = npz['cnorms']
    cnt, total_cnt = 0, 0
    
    with open(os.path.join(anno_path, '%s.txt'%data), mode='r') as f, \
        open(os.path.join(anno_path, '%s.lon'%data), mode='r') as sqlfile:
        random.seed(time.time())
        samples = [random.randint(0,37713-1) for _ in range(10)]
        symlines = symfile.readlines()
        lines = f.readlines()
        sqls = sqlfile.readlines()
        print(' # sqls {0}, # symbols {1}, # questions {2}, # wnorms {3}, # cnorms {4} '.format(len(sqls), len(symlines), len(lines), len(wnorms), len(cnorms)))
        for i, (line, sym_line, sql) in enumerate(zip(lines, symlines, sqls)):
            if i >= 100992:
                break
            arr = wnorms[i]
            carr = cnorms[i]
            if line.strip('\n').endswith('1'):
                total_cnt += 1
                qu = line.split('\t')[0]
                if total_cnt in samples:
                    y = arr
                    g = carr
                    x = qu.split()
                    y = arr[:len(x)]
                    g = carr[:len(x)]*10
                    xi = [i for i in range(0, len(x))]
                    ax = fig.add_subplot(gs[samples.index(total_cnt), :])
                    ax.plot(xi, y, label='word level')
                    ax.plot(xi, g, color='r', label='char level')
                    ax.set_xticks(xi)
                    ax.set_xticklabels(x, fontsize=8, rotation=30)
                    ax.set_xlabel(sql)
                    ax.legend(loc='upper left')

    
    plt.savefig(file_name)
