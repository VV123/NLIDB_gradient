from __future__ import print_function
import sys
import os
import keras
import copy
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from utils.data_manager import load_data,load_vocab_all
from collections import defaultdict
from argparse import ArgumentParser

import sys
reload(sys)
sys.setdefaultencoding('utf8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# ----------------------------------------------------------------------------
_PAD = 0
_GO = 1
_END = 2
subset='all'
path = os.path.abspath(__file__)
annotation_path = os.path.dirname(path) + '/data/DATA/wiki/'
overnight_path = os.path.dirname(path) + '/data/DATA/overnight/'
# ----------------------------------------------------------------------------
def decode_data_recover(sess, env, X_data, y_data, s, batch_size=128):
    """
    Inference and calculate EM acc based on recovered SQL
    """
    print('\nDecoding and Evaluate recovered EM acc')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    inf_logics = [] 
    _, reverse_vocab_dict, _, _ = load_vocab_all()
    acc = 0
    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch+1, n_batch),end='\r')
        sys.stdout.flush()
        start = batch * batch_size
        end = min(n_sample, start+batch_size)
        cnt = end - start
        ybar = sess.run(env.pred_ids,
                        feed_dict={env.x: X_data[start:end]})
        ybar = np.squeeze(ybar[:,0,:])# pick top prediction
        for seq in ybar:
            seq = list(seq)
            seq.append(_END)
            seq=seq[:seq.index(_END)]
            logic=" ".join([reverse_vocab_dict[idx] for idx in seq])
            inf_logics.append(logic) 
     
    xtru, ytru = X_data, y_data
    with gfile.GFile(annotation_path+'%s_infer.txt'%s, mode='w') as output, \
        gfile.GFile(annotation_path+'%s_ground_truth.txt'%s, mode='r') as S_ori_file, \
        gfile.GFile(annotation_path+'%s_sym_pairs.txt'%s, mode='r') as sym_pair_file:

        sym_pairs = sym_pair_file.readlines()  # annotation pairs from question & table files
        S_oris = S_ori_file.readlines()  # SQL files before annotation
        for true_seq, logic, x, sym_pair, S_ori in zip(ytru, inf_logics, xtru, sym_pairs, S_oris):
            sym_pair = sym_pair.replace('<>\n','')
            S_ori = S_ori.replace('\n','')
            Qpairs = [pair.split('=>') for pair in sym_pair.split('<>')]
            true_seq = list(true_seq[1:])    # delete <eos>
            x = list(x[1:])   # delete <eos>
            true_seq.append(_END)
            true_seq=true_seq[:true_seq.index(_END)]
            x.append(_END)
            x=x[:x.index(_END)]
            
            xseq = " ".join([reverse_vocab_dict[idx] for idx in x])
            true_logic = " ".join([reverse_vocab_dict[idx] for idx in true_seq])

            logic = logic.replace(' (','').replace(' )','')
            true_logic = true_logic.replace(' (','').replace(' )','') 

            recover = logic
            for sym, word in Qpairs:
                recover = recover.replace(sym, word)
            output.write(recover + '\n')

            logic_tokens = logic.split()
            if __switch_cond(logic, true_logic):
                logic = true_logic

            recover_S = logic
            for sym, word in Qpairs:
                recover_S = recover_S.replace(sym, word) 
            if __switch_cond(recover_S, S_ori):
                recover_S = S_ori
                
            acc += (recover_S==S_ori)
            #output.write(recover_S + '\n')
            
    print('EM: %.4f'%(acc*1./len(y_data)))  
    print('number of correct ones:%d'%acc)
    
    return acc*1./len(y_data)

def decode_data_recover_overnight(sess, env, X_data, y_data, subset, batch_size=128):
    """
    Inference and calculate EM acc based on recovered SQL
    """
    print('\nOVERNIGHT: Decoding and Evaluate recovered EM acc')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    true_values , values, inf_logics = [], [], []
    _, reverse_vocab_dict, _, _ = load_vocab_all()
    i, acc = 0, 0
    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch+1, n_batch),end='\r')
        sys.stdout.flush()
        start = batch * batch_size
        end = min(n_sample, start+batch_size)
        cnt = end - start
        ybar = sess.run(env.pred_ids,
                        feed_dict={env.x: X_data[start:end]})
        ybar = np.asarray(ybar)
        ybar = np.squeeze(ybar[:,0,:])  # pick top prediction
        for seq in ybar:
            try:
                seq=seq[:list(seq).index(_END)]
            except ValueError:
                pass  
            logic=" ".join([reverse_vocab_dict[idx] for idx in seq])
            inf_logics.append(logic) 
     
    xtru, ytru = X_data, y_data
    with gfile.GFile('%s.txt'%subset, mode='w') as output, gfile.GFile(overnight_path+'%s/%s_ground_truth.txt'%(subset,subset), mode='r') as S_ori_file, \
        gfile.GFile(overnight_path+'%s/%s_sym_pairs.txt'%(subset,subset), mode='r') as sym_pair_file:

        sym_pairs = sym_pair_file.readlines()  # annotation pairs from question & table files
        S_oris = S_ori_file.readlines()  # SQL files before annotation
        for true_seq, logic, x, sym_pair, S_ori in zip(ytru, inf_logics, xtru, sym_pairs, S_oris):
            sym_pair = sym_pair.replace('<>\n','')
            S_ori = S_ori.replace('\n','').replace(' (','').replace(' )','')
            Qpairs = []
            if sym_pair != '\n':
                for pair in sym_pair.split('<>'):
                    Qpairs.append(pair.split('=>'))
            true_seq = true_seq[1:]    # delete <eos>
            x = x[1:]   # delete <eos>
            try:
                true_seq=true_seq[:list(true_seq).index(_END)]
            except ValueError:
                pass

            try:
                x=x[:list(x).index(_END)]
            except ValueError:
                pass
            
            xseq = " ".join([reverse_vocab_dict[idx] for idx in x])
            true_logic = " ".join([reverse_vocab_dict[idx] for idx in true_seq])

            logic = logic.replace(' (','').replace(' )','')
            true_logic = true_logic.replace(' (','').replace(' )','') 

            logic_tokens = logic.split()
            if __switch_cond(logic, true_logic):    
                logic = true_logic

            recover_S = logic
            if Qpairs != []:
                for sym, word in Qpairs:
                    recover_S = recover_S.replace(sym, word) 

            logic_tokens = recover_S.split()
            if __switch_cond(recover_S, S_ori):    
                recover_S = true_logic

            acc += (recover_S==S_ori)
            #output.write(recover_S + '\n')
            
            i += 1
            true_values.append(true_logic)
            values.append(logic)        
    
    print('EM: %.4f'%(acc*1./len(y_data)))  
    print('number of correct ones:%d'%acc)
    
    return acc, len(y_data)

def decode_one(sess, env, X_data, batch_size=1):
    print('\nDecoding one')
    _,reverse_vocab_dict,_,_=load_vocab_all()

    ybar = sess.run(
        env.pred_ids,
        feed_dict={env.x: X_data})
    xtru = X_data
    ybar = np.asarray(ybar)
    ybar = np.squeeze(ybar[:,0,:])

    if len(xtru) != len(ybar):
        ybar = [ybar]
    #print(xtru)
    #print(ybar)
    for seq, x in zip(ybar, xtru):
        x = x[1:]
        seq=list(seq)
        x=list(x)
        seq.append(_END)
        seq=seq[:list(seq).index(_END)]
        x.append(_END)
        x=x[:list(x).index(_END)]

        xseq = " ".join([reverse_vocab_dict[idx] for idx in x ])
        logic = " ".join([reverse_vocab_dict[idx] for idx in seq ])
        print(xseq)
        print(logic)
        
def decode_data(sess, env, X_data, y_data , batch_size=128):
    """
    Inference w/o recover annotation symbols
    """
    print('\nDecoding')
    n_sample = X_data.shape[0]
    sample_ids = np.random.choice(n_sample, 100)
    n_batch = int((n_sample+batch_size-1) / batch_size)
    acc = 0
    true_values , values = [], []
    _,reverse_vocab_dict,_,_=load_vocab_all()
    with gfile.GFile('output.txt', mode='w') as output:
        i = 0
        for batch in range(n_batch):
            print(' batch {0}/{1}'.format(batch+1, n_batch),end='\r')
            sys.stdout.flush()
            start = batch * batch_size
            end = min(n_sample, start+batch_size)
            cnt = end - start
            ybar = sess.run(
                env.pred_ids,
                feed_dict={env.x: X_data[start:end]})
            xtru = X_data[start:end]
            ytru = y_data[start:end]
            ybar = np.asarray(ybar)
            ybar = np.squeeze(ybar[:,0,:])
            for true_seq,seq,x in zip(ytru, ybar, xtru):
                true_seq = true_seq[1:]
                x = x[1:]
                
                try:
                    true_seq=true_seq[:list(true_seq).index(_END)]
                except ValueError:
                    pass
                try:
                    seq=seq[:list(seq).index(_END)]
                except ValueError:
                    pass
                try:
                    x=x[:list(x).index(_END)]
                except ValueError:
                    pass
                
                xseq = " ".join([reverse_vocab_dict[idx] for idx in x ])
                logic = " ".join([reverse_vocab_dict[idx] for idx in seq ])
                true_logic = " ".join([reverse_vocab_dict[idx] for idx in true_seq ])

                logic = logic.replace(' (','').replace(' )','')
                true_logic = true_logic.replace(' (','').replace(' )','') 
                logic_tokens = logic.split()
                
                if __switch_cond(logic, true_logic):
                    logic = true_logic
                acc += (logic==true_logic)
                i += 1
    print('EM: %.4f'%(acc*1./len(y_data)))  
    print('number of correct ones:%d'%acc)
    
    return acc*1./len(y_data) 
#----------------------------------------------------------------------
'''
def __switch_cond(logic_tokens, true):
    def _parse(logic_tokens):
        try:
            where_idx = logic_tokens.index('where')
            where_clause = logic_tokens[:where_idx+1]
        except:
            where_clause = None
            where_idx = -1
        logic_tokens = logic_tokens[where_idx+1:]

        and_clause = []
        while 'and' in logic_tokens:
            and_idx = logic_tokens.index('and')
            and_clause.append(logic_tokens[:and_idx])
            logic_tokens = logic_tokens[and_idx+1:]
        if logic_tokens:
            and_clause.append(logic_tokens)
        #print(where_clause)
        #print(and_clause)
        return where_clause, and_clause
    
    where1, and1 = _parse(logic_tokens)
    where2, and2 = _parse(true.split())

    if where1 == None and where2 == None and and1 == [] and and2 == []:
        return logic_tokens == true.split()

    if where1 != where2 or len(and1) != len(and2):
        return False
    and1_copy = copy.copy(and1)
    for cond in and1_copy:
        if cond in and2:
            and2.remove(cond)
            and1.remove(cond)
    return and1 == [] and and2 == []
'''

def __switch_cond(q1, q2):
    if 'where' not in q1 or 'where' not in q2:
        return False
    q1 = ' '.join(q1.split())
    q2 = ' '.join(q2.split())
    prefix1 = q1[:q1.index('where') + 5]
    prefix1 = ' '.join(prefix1.split())
    q1 = q1[q1.index('where') + 5: ]
    prefix2 = q2[:q2.index('where') + 5]
    prefix2 = ' '.join(prefix2.split())
    if prefix1 != prefix2:
        return False
    q2 = q2[q2.index('where') + 5: ]
    conds1 = q1.split(' and ')
    conds2 = q2.split(' and ')
    conds1 = [ ' '.join(cond.split()) for cond in conds1]
    conds2 = [ ' '.join(cond.split()) for cond in conds2]
    conds1 = sorted(conds1, key=lambda x : x)
    conds2 = sorted(conds2, key=lambda x : x)
    return conds1 == conds2


if __name__ == '__main__':
    a = 'round where pole position equal 1-1 and grand prix equal mario cipollini and winning constructor equal norwich city'
    b = 'round where pole position equal 1-1 and winning constructor equal norwich city and grand prix equal mario cipollini'
    re = __switch_cond(a, b)
    print(re)
