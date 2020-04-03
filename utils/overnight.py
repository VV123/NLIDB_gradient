import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from utils import glove
from collections import defaultdict
embedding_dim = glove.Glove.embedding_dim
path = '/home/wzw0022/ww_text/data/overnight'
all_path = '/home/wzw0022/ww_text/data/overnight/all'
save_path = '/home/wzw0022/Wiki_contrib/data/overnight'
_PAD = 0
_GO = 1
_END = 2
def build_vocab_all(load=True,data_file1=os.path.join(all_path,'train.lon'),data_file2=os.path.join(all_path,'train.qu')):
    if load==False:
        vocabs = set()
        with gfile.GFile(data_file1,mode='r') as DATA:
            lines = DATA.readlines()
            for line in lines:
                for word in line.split():
                    if word not in vocabs:
                        vocabs.add(word)
        with gfile.GFile(data_file2,mode='r') as DATA:
                        lines = DATA.readlines()
                        for line in lines:
                                for word in line.split():
                                        if word not in vocabs:
                                                vocabs.add(word)
        vocab_tokens = ['pad','bos','eos']
        vocab_tokens.extend(list(vocabs))
        np.save('vocab_tokens_all.npy',vocab_tokens)
        
    else:
        vocab_tokens=np.load('vocab_tokens_all.npy')
    return vocab_tokens


def load_vocab_all(load=True):
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
        np.save('vocab_dict_all.npy',vocab_dict)
        np.save('reverse_vocab_dict_all.npy',reverse_vocab_dict)
        vocab_emb = embedding.embedding(vocab_tokens, maxlen=1)
        vocab_emb = vocab_emb[:,0] #discard begin word
        np.save('vocab_emb_all.npy',vocab_emb)
        print('vocab shape:')
        print(vocab_emb.shape)  
    else:
       vocab_dict=np.load('vocab_dict_all.npy').item()
       reverse_vocab_dict=np.load('reverse_vocab_dict_all.npy').item()
       vocab_emb = np.load('vocab_emb_all.npy')
    return vocab_dict,reverse_vocab_dict,vocab_emb


def load_data_idx(maxlen=20,subset='basketball',load=True,s='train'):
    all_q_tokens = []
    all_logic_ids = []
    vocab_dict,_,_=load_vocab_all()
    vocab_dict = defaultdict(lambda:'unk',vocab_dict)
    questionFile=os.path.join(path,'%s/%s.qu'%(subset,s))
    logicFile=os.path.join(path,'%s/%s.lon'%(subset,s))
    with gfile.GFile(questionFile, mode='r') as questions, gfile.GFile(logicFile, mode='r') as logics:
        q_sentences = questions.readlines()
        logics = logics.readlines()
        assert len(q_sentences)==len(logics)
        for q_sentence,logic in zip(q_sentences,logics):
            token_ids = [_GO]
            token_ids.extend([vocab_dict[x] for x in q_sentence.split()])
            token_ids.append(_END)
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
    return all_q_tokens,all_logic_ids

#load_data_idx(subset='restaurants',load=False,s='train')
