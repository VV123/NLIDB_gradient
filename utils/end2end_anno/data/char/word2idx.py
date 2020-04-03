import tensorflow as tf
import numpy as np
import os
path = os.path.abspath(__file__)
savepath = os.path.dirname(path).replace('/char','')
datapath = os.path.dirname(path).replace('utils/end2end_anno/data/char','data/end2end')
import codecs
import nltk
import pickle
import copy
import itertools
"""
Padded char to index
"""
vocab_size = 1224
def word2idx(maxlen_p=60,maxlen_q=3,maxwlen=10,save=False):
    filepath ='wiki_word.npz'
    filepath_X = os.path.expanduser(os.path.join(savepath, filepath))
    filepath = 'wiki_label.npz'
    filepath_y = os.path.expanduser(os.path.join(savepath, filepath))

    
    tkr = tf.keras.preprocessing.text.Tokenizer(num_words=None,
                                            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                            lower=True, split=" ", char_level=True)
    
     
    with open(os.path.join(os.path.dirname(path), 'tokenizer.pickle'), 'rb') as handle:
        tkr = pickle.load(handle) 
    #tkr.word_index[' '] = vocab_size + 1
    #tkr.word_index['^'] = vocab_size + 2
    #tkr.word_index['$'] = vocab_size + 3
    #_PAD = tkr.word_index[' '] 
    #_SPACE = tkr.word_index[' '] 
    #_BOT = tkr.word_index['^'] 
    #_EOT = tkr.word_index['$']
    _PAD = vocab_size + 1
    _SPACE = vocab_size + 1
    _BOT = vocab_size + 2
    _EOT = vocab_size + 3

    #reverse_word_map = dict(map(reversed, tkr.word_index.items()))
    #print(reverse_word_map.keys())
    #print(reverse_word_map[47])

    def _seq2idx(texts, maxlen, maxwlen, bos=1):
        maxlen -= bos # bos
        print('\nAllocating index matrix')
        vec = np.tile(_PAD, (len(texts) * (maxlen+bos)*(maxwlen+2+1), 1))
        vec = np.reshape(vec, (len(texts), (maxlen+bos)*(maxwlen+2+1)))
        vec = vec.astype(np.int)
        print(len(texts))
        print(vec.shape)

        print('\nDo char to index...')
        for i, text in enumerate(texts):
            text = text.strip(' ')
            text = text.split(' ')

            idxs = tkr.texts_to_sequences(text)
            pad_idxs = []
            if len(idxs) >= maxlen:
                idxs = idxs[:maxlen]
                
            for w_idx in idxs:
                if len(w_idx) >= maxwlen:
                    padded = np.asarray(w_idx[:maxwlen])
                else:
                    padded = np.pad(np.asarray(w_idx), (0,maxwlen-len(w_idx)), 'constant', constant_values=(0,_PAD))
                padded = np.pad(padded, (1,1), 'constant', constant_values=(_BOT,_EOT))
                padded = np.append(padded, _SPACE)
                pad_idxs.append(padded)
            pad_idxs = list(itertools.chain.from_iterable(pad_idxs))
            if bos:
                pad_idxs.extend([_EOT]*(maxwlen + 2 + 1))
            vec[i][:len(pad_idxs)] = pad_idxs
        return vec	

    def _char2idx(fpath):
        for line in codecs.open(fpath,'r','utf-8-sig'):
            assert len(line.split('\t')) == 3 or line.startswith('#')
        questions = [ line.split('\t')[0] for line in codecs.open(fpath,'r','utf-8-sig') if not line.startswith('#')]
        print('# of questions:' + str(len(questions)))
        cols = [ line.split('\t')[1] for line in codecs.open(fpath,'r','utf-8-sig') if not line.startswith('#')]
        print('# of cols:' + str(len(cols)))
        labels = [ line.split('\t')[2] for line in codecs.open(fpath,'r','utf-8-sig') if not line.startswith('#')]
        print('# of labels:' + str(len(labels)))
        questions_idx = _seq2idx(questions, maxlen_p, maxwlen, bos=1)
        cols_idx = _seq2idx(cols, maxlen_q, maxwlen, bos=1)

        return questions_idx, cols_idx, labels


    print('\nGenerating training/test data')
    print(len(tkr.word_index))
    X_train0, X_train1, y_train = _char2idx(os.path.join(datapath, 'train_model.txt'))
    X_test0, X_test1, y_test = _char2idx(os.path.join(datapath, 'test_model.txt'))
    X_dev0, X_dev1, y_dev = _char2idx(os.path.join(datapath, 'dev_model.txt'))
    print('\nSaving...')
    np.savez(filepath_X, X_train0=X_train0, X_test0=X_test0, X_dev0=X_dev0, X_train1=X_train1, X_test1=X_test1, X_dev1=X_dev1)
    np.savez(filepath_y, y_train=y_train, y_test=y_test, y_dev=y_dev)
    print('\nSaved!')
if __name__ == "__main__":
    word2idx()
