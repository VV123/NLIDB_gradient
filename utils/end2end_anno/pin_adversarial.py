import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from numpy import linalg as LA
import copy
import time
import scipy
import re
import pickle
import sys
#sys.path.append('/home/wzw0022/match-lstm/annotation/lib')
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

def _pin_adv(data='test', anno_path='data/annotated'): 
   
    fig = plt.figure(figsize=(10, 20))
    gs = gridspec.GridSpec(10, 10, wspace=0.1, hspace=1)
    st = fig.suptitle("char&word model", fontsize="x-large") 
    file_name = 'pin_%s_gradient.pdf'%data
    assert data != 'train'
    gradient_path = '/nfs_shares/wzw0022_home/%s_gradient_norm_conv1.npz'%data
    npz = np.load(gradient_path)
    wnorms = npz['wnorms']
    cnorms = npz['cnorms']
    ybar = npz['ybar']
    cnt, total_cnt, match_cnt = 0, 0, 0
    head_cnt, pair_cnt = 0, 0
    Q_anno_cnt = 0
    cnt1 = 0 
    F2V, V2F = _load_dict(split=data, anno_path=anno_path)
    with open(os.path.join(anno_path, '%s.ori.qu'%data), mode='r') as ori_que_file,\
            open(os.path.join(anno_path, '%s.ori.lon'%data), mode='r') as ori_lon_file:
        qus = ori_que_file.readlines()
        lons = ori_lon_file.readlines()
        assert len(qus) == len(lons)
        print('Number of records: %d'%len(qus))
    with open(os.path.join(anno_path, '%s.txt'%data), mode='r') as f,\
            open(os.path.join(anno_path, '%s.lon'%data), mode='r') as sqlfile,\
            open(os.path.join(save_path, '%s_gradient_sym_pairs.txt'%data), mode='w') as pair_file,\
            open(os.path.join(save_path, '%s_head.txt'%data), mode='w') as head_file:
        random.seed(time.time())
        samples = [random.randint(0,37713-1) for _ in range(1000)]
        plot_num = 0
        lines = f.readlines()
        sqls = sqlfile.readlines()
        START = ''
        question = START
        cand_pairs = [] #init
        gradient_cand_all = []
        ii = 0
        multi_heads = 0
        for _, (line, sql) in enumerate(zip(lines, sqls)):
            #print('{0}/{1}'.format(i, len(lines)), end='\r')
            if line.startswith('#') and sql.startswith('#'):
                """store""" 
                if question != START:
                    cand_pairs = sorted(cand_pairs, key=lambda x:x[1])
                    '''Use v2f to match more'''
                    Vs = sorted(v2f.keys(), key=lambda x: len(x), reverse=True)
                    matched_vs = [v for _, v in cand_pairs]
                    for V in Vs:
                        if len(V) > 2 and not any([V in v for v in matched_vs]) and V in ori_question:
                            column = v2f[V][0]
                            cand_pairs.append([column, V])
                    ''''''
                    cand_pairs = sorted(cand_pairs, key=lambda x:x[1])
                    gradient_cand_all.append(cand_pairs)
                    cand_pairs = []
                    '''Pick head'''
                    head = ''
                    if HEADS:  
                        if len(HEADS)>1:
                            multi_heads += 1
                        
                        (pin, head) = sorted(HEADS, key=lambda x: (qu.index(x[0]), -len(x[1])))[0]
                    if head != '':
                        head_file.write(head+'<><>'+pin+'\n')
                    else:
                        head_file.write('\n')
                    ''''''
                   
                """reset"""
                qu_idx = int(line.strip('\n').strip('#'))
                if qu_idx < len(qus):
                    question = qus[qu_idx]
                    ori_question = question
                    f2v, v2f = F2V[qu_idx], V2F[qu_idx]
                    HEADS = []
                continue 
            
            pred_label = ybar[ii]
            true_label = int(line.strip('\n').split('\t')[2])
            if pred_label > .5:
                total_cnt += 1
                qu = line.split('\t')[0]
                pin, idx = _pin_mountain(qu, wnorms[ii])
                
                delimiter = '~~'
                line = line.replace('\u2003',' ')
                true_col = line.split('\t')[1].replace(' <bos>','').replace(' <pad>','')
                assert qus[qu_idx].strip('\n') == qu
                qu = qu.replace('\u2009',' ').replace(u'\xa0',u' ')
                vals = f2v[true_col]
                assert len(vals)>0

                pin_vals = [_in(val,qu) for val in vals if _in(val,qu)]
                pin_vals = sorted(pin_vals, key=lambda x:len(x), reverse=True)
                """
                Write annotated question and sql
                """
                if len(pin_vals)>0:
                    val = pin_vals[0]
                    cand_pairs.append([true_col, val])
                else:#HEAD
                    HEADS.append((pin, true_col))
            
            ii += 1

    with open (os.path.join(save_path, '%s_cand_pairs.pkl'%data), 'wb') as fp:
        pickle.dump(gradient_cand_all, fp)

def __isnum(val):
    return re.match("^\-?\+?\d+\.?\d*$", val) or re.match("^\-?\+?\d*\.?\d+$", val) or re.match("^\d+e\d+$", val)

def __token_isnum(token):
    if __isnum(token):
        return token
    elif re.match("^\$?\+?\d+\.?\d*[^d0mc&\(⁄\/\.×x:\+–-]{1,3}\+?\.?$", token) or re.match("^\$?\+?\d*\.?\d+[^d0mc&\(⁄\/\.×x:\+–-]{1,3}\+?\.?$", token):
        number = [ s for s in re.findall("\d*\.?\d*", token) if s!='']
        if '2' in number and len(number) > 1:
            number.remove('2')
        if '.' in number and len(number) > 1:
            number.remove('.')
        assert len(number)==1
        number = number[0]
        return number
    else:
        return None

def _in(val, qu):
    if __isnum(val):
        for token in qu.split():
            if __token_isnum(token) and abs(float(__token_isnum(token))-float(val))<1e-5:
                return token
        return None
    elif val in qu:
        return val
    return None

def _pin_single(line, norms):
    line = copy.copy(line)
    norms = copy.copy(norms)
    maxidx = np.argmax(norms)
    maxnorm = norms[maxidx]
    del norms[maxidx]
    second = np.argmax(norms)
    if (maxnorm - second)/maxnorm > .5:
        return True
    return False

def _pin_mountain(line, norms):
    tokens = line.strip('\n').split()
    norms = copy.copy(norms[:len(tokens)]).tolist()
    if _pin_single(line, norms):
        return tokens[np.argmax(norms)], np.argmax(norms) 
    pin, idx = [], []
    pre, pre_norm = -1, None
    i = 0
    idxs = sorted(range(len(norms)), key=lambda x: norms[x], reverse=True)
    while len(pin)<3 and len(norms)>0 and i<len(norms):
        top = idxs[i]
        if pre < 0 or abs(top-pre)==1:
            pin.append(tokens[top])
            idx.append(top)
            pre = top
            pre_norm = norms[top]
        i += 1
    idx = sorted(idx)
    tokens = np.asarray(line.strip('\n').split())[idx].tolist()
    return ' '.join(tokens), idx
        

def _pin_percent(line, norms, percent=.9):
    tokens = line.strip('\n').split()
    norms = norms[:len(tokens)]
    sorted_index = np.argsort(norms)

    tokens = np.asarray(tokens)
    #sorted(range(len(tokens)), key=lambda k: tokens[k])
    assert len(tokens) == len(norms)
 
    index = sorted_index[int(percent*len(norms)):]
    return ' '.join(tokens[index].tolist()), index

def _load_dict(split='train', anno_path=''):
    path = os.path.join(anno_path,'%s_dict.npz'%split)
    data = scipy.load(path)
    f2v = data['f2v_all']
    v2f = data['v2f_all']
    return f2v, v2f

if __name__ == '__main__':
    for dataset in ['test', 'dev']:
        print('------------'+dataset+'--------------')
        _pin_adv(dataset, save_path)
