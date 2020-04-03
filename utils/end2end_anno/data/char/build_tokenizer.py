import tensorflow as tf
import numpy as np
import os
path = os.path.abspath(__file__)
datapath = os.path.dirname(path).replace('utils/end2end_anno/data/char','data/end2end')
import codecs
import nltk
import pickle
import copy
import itertools


def char2idx(maxlen_p=300,maxlen_q=30):

    tkr = tf.keras.preprocessing.text.Tokenizer(num_words=None,
                                            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                            lower=True, split=" ", char_level=True)
    def _seq2idx(texts, maxlen, bos=1):
        for i, text in enumerate(texts):
            tkr.fit_on_texts(text)	

    def _char2idx(fpath):
        for line in codecs.open(fpath,'r','utf-8-sig'):
            assert len(line.split('\t')) == 3
        questions = [ line.split('\t')[0] for line in codecs.open(fpath,'r','utf-8-sig') ]
        cols = [ line.split('\t')[1] for line in codecs.open(fpath,'r','utf-8-sig') ]
        labels = [ line.split('\t')[2] for line in codecs.open(fpath,'r','utf-8-sig') ]
        questions_idx = _seq2idx(questions, maxlen_p-1, bos=1)
        cols_idx = _seq2idx(cols, maxlen_q-1, bos=1)

        return questions_idx, cols_idx, labels


    print('\nGenerating training/test data')
    #with open('/home/wzw0022/match-lstm/data/tokenizer.pickle', 'rb') as handle:
    #    tkr = pickle.load(handle)
    X_train0, X_train1, y_train = _char2idx(os.path.join(datapath, 'train_model.txt'))
    X_test0, X_test1, y_test = _char2idx(os.path.join(datapath, 'test_model.txt'))
    X_dev0, X_dev1, y_dev = _char2idx(os.path.join(datapath, 'dev_model.txt'))
    print(len(tkr.word_index))
    with open('token.pickle', 'wb') as handle:
        pickle.dump(tkr, handle, protocol=pickle.HIGHEST_PROTOCOL)
if __name__ == "__main__":
	#char2idx()
	with open('/home/wzw0022/match-lstm/data/tokenizer.pickle', 'rb') as handle:
		tkr = pickle.load(handle)
	with open('token.pickle', 'rb') as handle:
		tkr1 = pickle.load(handle)	

	keys = tkr.word_index.keys()
	keys1 = tkr1.word_index.keys()
	c = 0
	for i in keys1:
		if tkr.word_index[i] != tkr1.word_index[i]:
			
			print('----')
			print(tkr.word_index[i])
			print(tkr1.word_index[i])
	print(c)
	#print(tkr.word_index)
	#print(sorted(tkr.word_index.keys()))
	#print(tkr1.word_index)
	#print(sorted(tkr1.word_index.keys()))
