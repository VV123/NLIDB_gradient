from __future__ import print_function
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import glove
embedding_dim = glove.Glove.embedding_dim
path='/home/wzw0022/DATA/WikiSQL'
all_path = '/home/wzw0022/DATA/WikiSQL'
'''
0: pad
1: bos
2: eos
'''
_PAD = 0
_GO = 1
_END = 2
def build_vocab_all( load=True, files=['train.lon','train.qu'] ):
	if load==False:
		vocabs = set()
		for fname in files:
			fname = os.path.join( path, fname )
			with gfile.GFile(fname, mode='r') as DATA:
				lines = DATA.readlines()
				for line in lines:
					for word in line.split():
						if word not in vocabs:
							vocabs.add(word)
		vocab_tokens = ['pad','bos','eos']
                vocab_tokens.extend(list(vocabs))
		np.save('data/vocab_tokens_all.npy',vocab_tokens)
		
	else:
		vocab_tokens=np.load('data/vocab_tokens_all.npy')
	return vocab_tokens

def load_vocab_all( load=True ):

    if load==False:
        vocab_dict = {}
        reverse_vocab_dict = {}
        embedding = glove.Glove()
        vocabs = build_vocab_all()
        vocab_tokens = []
        for i,word in enumerate(vocabs):
            vocab_dict[word]=i
            reverse_vocab_dict[i]=word
            vocab_tokens.append([word])
        np.save('data/vocab_dict_all.npy',vocab_dict)
        np.save('data/reverse_vocab_dict_all.npy',reverse_vocab_dict)
        vocab_emb = embedding.embedding(vocab_tokens, maxlen=1)
        vocab_emb = vocab_emb[:,0] #retrieve embedding
        np.save('data/vocab_emb_all.npy',vocab_emb)
        print('Vocab shape:')
        print(vocab_emb.shape)
    else:
        vocab_emb=np.load('data/vocab_emb_all.npy')
        vocab_dict=np.load('data/vocab_dict_all.npy').item()
        reverse_vocab_dict=np.load('data/reverse_vocab_dict_all.npy').item()
        print('Vocab shape:')
        print(vocab_emb.shape)
    return vocab_dict,reverse_vocab_dict,vocab_emb


def load_data_idx(maxlen=20,subset='basketball',load=True,s='train'):
	if load:
		emb = np.load('data/vocab_emb_all.npy')
		print('========embedding shape========')
		print(emb.shape)
		all_q_tokens = np.load('data/%s_qu_idx.npy'%(s))
		all_logic_ids = np.load('data/%s_lon_idx.npy'%(s))
	else:	
		all_q_tokens = []
		all_logic_ids = []
		vocab_dict,_=load_vocab_all()
		questionFile=os.path.join(path,'%s.qu'%(s))
		logicFile=os.path.join(path,'%s.lon'%(s))
		with gfile.GFile(questionFile, mode='r') as questions, gfile.GFile(logicFile, mode='r') as logics:
			q_sentences = questions.readlines()
			logics = logics.readlines()
			assert len(q_sentences)==len(logics)
			i = 0
			length = len(logics)
			for q_sentence,logic in zip(q_sentences,logics):
				i+=1
				print('counting: %d / %d'%(i,length),end='\r')
				sys.stdout.flush()
				#for x in q_sentence.split():
				#	if x not in vocab_dict.keys():
				#		print('QUESTION ERROR:not in dict:%s'%x)
				token_ids = [_GO]
				token_ids.extend([vocab_dict[x] for x in q_sentence.split()])
				token_ids.append(_END)
				#for x in logic.split():
				#	if x not in vocab_dict.keys():
				#		print("LOGIC ERROR:not found in dict:%s"%x)
				logic_ids = [_GO]
				logic_ids.extend([vocab_dict[x] for x in logic.split()])
				logic_ids.append(_END)
				if maxlen>len(logic_ids):
					logic_ids.extend([ _PAD for i in range(len(logic_ids),maxlen)])
				else:
					logic_ids = logic_ids[:maxlen]
				if maxlen>len(token_ids):
					token_ids.extend([ _PAD for i in range(len(token_ids),maxlen)])
				else:
					token_ids = token_ids[:maxlen]
				all_q_tokens.append(token_ids)
				all_logic_ids.append(logic_ids)
		all_logic_ids=np.asarray(all_logic_ids)
		print(all_logic_ids.shape)
		all_q_tokens=np.asarray(all_q_tokens)
		np.save('data/%s_lon_idx.npy'%s,all_logic_ids)
		np.save('data/%s_qu_idx.npy'%s,all_q_tokens)
	return all_q_tokens,all_logic_ids

load_data_idx(load=True,s='train')
