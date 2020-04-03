#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from annotation import glove
from collections import defaultdict

embedding_dim = 300
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path).replace('utils', 'data')
wiki_path = dir_path + '/DATA/wiki'
save_path = dir_path
'''
0: pad
1: bos
2: eos
'''
_PAD = 0
_GO = 1
_END = 2
_EOF = 3
_UNK = 4
_TOKEN_NUMBER = 5
_TOKEN_MODEL = 6
_EOC = 7
ori_files1 = [ 'train.lon', 'train.qu', 'test.lon', 'test.qu', 'dev.lon', 'dev.qu']
ori_files2 = [ 'train_vocab.lon', 'train_vocab.qu', 'test_vocab.lon', 'test_vocab.qu']
vocab_files = [ os.path.join(wiki_path, x) for x in ori_files1 ]
for subset in ['basketball', 'calendar', ...]:
    overnight_path = dir_path + '/DATA/overnight/%s'%subset
    vocab_files.extend([ os.path.join(overnight_path, x) for x in ori_files2 ])
annotation = ['<f0>','<f1>','<f2>','<f3>','<v0>','<v1>','<v2>','<v3>']
def build_vocab_all(load=True, files=vocab_files):
    if load==False:
        print('\nRebuild vocab...')
        vocab_tokens = ['<pad>','<bos>','<eos>','<eof>','<unk>','<@number>','<@model>','<eoc>']
        vocab_tokens.extend(annotation)
        vocabs = set()
        
        for fname in files:
            with gfile.GFile(fname, mode='r') as DATA:
                lines = DATA.readlines()
                for line in lines:
                    for word in line.split():
                        if word not in vocabs and word not in vocab_tokens:
                            vocabs.add(word)
        
        vocab_tokens.extend(list(vocabs))
        np.save(os.path.join(save_path,'vocab_tokens_all.npy'),vocab_tokens)
        print('\nBuild vocab done...')
        print('\nVocab size: %d'%len(vocab_tokens))
    else:
        vocab_tokens=np.load(os.path.join(save_path,'vocab_tokens_all.npy'))
        print('Vocab size:{0}'.format(len(vocab_tokens)))
    return vocab_tokens

def load_vocab_all(load=True):

    if load == False:
        print('\nRebuild vocab embedding...')
        vocab_dict = {}
        reverse_vocab_dict = {}
        embedding = glove.Glove()
        vocabs = build_vocab_all()
        vocab_tokens = []
        for i,word in enumerate(vocabs):
            vocab_dict[word]=i
            reverse_vocab_dict[i]=word.decode('utf-8')
            vocab_tokens.append([word.decode('utf-8')])
        np.save(os.path.join(save_path,'vocab_dict_all.npy'),vocab_dict)
        np.save(os.path.join(save_path,'reverse_vocab_dict_all.npy'),reverse_vocab_dict)
        vocab_emb, unk_idx = embedding.embedding(vocab_tokens, maxlen=1)
        unk_idx = np.asarray(unk_idx)
        vocab_emb = vocab_emb[:,0] #retrieve embedding
        train_idx = unk_idx
        #train_idx = np.concatenate( (np.arange(15), unk_idx) )
        np.save(os.path.join(save_path,'train_idx.npy'),train_idx)
        i = 0
        emb_dict = {}
        emb_dict['f'] = (np.random.rand(embedding_dim/2)-.5)*2*np.sqrt(3)
        emb_dict['v'] = (np.random.rand(embedding_dim/2)-.5)*2*np.sqrt(3)
        emb_dict['c'] = (np.random.rand(embedding_dim/2)-.5)*2*np.sqrt(3)
        for token,emb in zip(vocab_tokens,vocab_emb):
            token = token[0]
            if len(token)>=4 and token[0]=='<' and token[3]=='>' and token[2].isdigit():
                if token[2] in emb_dict:
                    right = emb_dict[token[2]]
                else:
                    emb_dict[token[2]] = (np.random.rand(embedding_dim/2)-.5)*2*np.sqrt(3)
                    right = emb_dict[token[2]]
                re = np.concatenate((emb_dict[token[1]],right))
                assert re.shape==(300,)
                vocab_emb[i]=re
            elif len(token)>=5 and token[0]=='<' and token[4]=='>' and token[2:4].isdigit():
                if token[2:4] in emb_dict:
                    right = emb_dict[token[2:4]]
                else:
                    emb_dict[token[2:4]] = (np.random.rand(embedding_dim/2)-.5)*2*np.sqrt(3)
                    right = emb_dict[token[2:4]]
                re = np.concatenate((emb_dict[token[1]],right))
                assert re.shape==(300,)
                vocab_emb[i]=re 
            i += 1

        np.save(os.path.join(save_path,'vocab_emb_all.npy'),vocab_emb)
    else:
        vocab_emb=np.load(os.path.join(save_path,'vocab_emb_all.npy'))
        vocab_dict=np.load(os.path.join(save_path,'vocab_dict_all.npy')).item()
        reverse_vocab_dict=np.load(os.path.join(save_path,'reverse_vocab_dict_all.npy')).item()
        train_idx = np.load(os.path.join(save_path,'train_idx.npy'))
    return vocab_dict, reverse_vocab_dict, vocab_emb, train_idx


def load_data_wiki(maxlen=30, load=True, s='train'):
    if load:
        emb = np.load(os.path.join(save_path,'vocab_emb_all.npy'))
        all_q_tokens = np.load(os.path.join(save_path,'wiki/%s_qu_idx.npy'%(s)))
        all_logic_ids = np.load(os.path.join(save_path,'wiki/%s_lon_idx.npy'%(s)))
    else:
        all_q_tokens = []
        all_logic_ids = []
        vocab_dict,_,_,_=load_vocab_all()
        vocab_dict = defaultdict(lambda:_UNK, vocab_dict)
        questionFile=os.path.join(wiki_path,'%s.qu'%(s))
        logicFile=os.path.join(wiki_path,'%s.lon'%(s))
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
                token_ids = [_GO]
                token_ids.extend([vocab_dict[x] for x in q_sentence.split()])
                for x in q_sentence.split():
                    if vocab_dict[x]==_UNK:
                        print('ERROR unknow word in question:'+x)
                #token_ids.append(_END)
                logic_ids = [_GO]
                logic_ids.extend([vocab_dict[x] for x in logic.split()])
                for x in logic.split():
                    if vocab_dict[x]==_UNK:
                        print('ERROR unknow word in logic:'+x)
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
            all_q_tokens=np.asarray(all_q_tokens)
            np.save(os.path.join(save_path,'wiki/%s_lon_idx.npy'%s),all_logic_ids)
            np.save(os.path.join(save_path,'wiki/%s_qu_idx.npy'%s),all_q_tokens)
	
    return all_q_tokens,all_logic_ids

def load_data_overnight(maxlen=30, subset=subset, load=False, s='train'):
    overnight_path = dir_path + '/DATA/overnight/%s'%subset
    all_q_tokens = []
    all_logic_ids = []
    vocab_dict, _, _, _ = load_vocab_all()
    vocab_dict = defaultdict(lambda: _UNK, vocab_dict)
    questionFile = os.path.join(overnight_path, 'new_%s.qu'%(s))
    logicFile = os.path.join(overnight_path, 'new_%s.lon'%(s))
    with gfile.GFile(questionFile, mode='r') as questions, gfile.GFile(logicFile, mode='r') as logics:
        q_sentences = questions.readlines()
        logics = logics.readlines()
        assert len(q_sentences) == len(logics)
        for q_sentence, logic in zip(q_sentences, logics):
            token_ids = [_GO]
            token_ids.extend([vocab_dict[x] for x in q_sentence.split()])
            #token_ids.append(_END)
            for x in token_ids:
                if x == _UNK:
                    print('ERROR')
            logic_ids = [_GO]
            logic_ids.extend([vocab_dict[x] for x in logic.split()])
            logic_ids.append(_END)
            for x in logic_ids:
                if x == _UNK:
                    print('ERROR')
            if maxlen > len(logic_ids):
                logic_ids.extend([ _PAD for i in range(len(logic_ids),maxlen)])
            else:
                logic_ids = logic_ids[:maxlen]
            if maxlen > len(token_ids):
                token_ids.extend([ _PAD for i in range(len(token_ids),maxlen)])
            else:
                token_ids = token_ids[:maxlen]
            all_q_tokens.append(token_ids)
            all_logic_ids.append(logic_ids)
    all_logic_ids = np.asarray(all_logic_ids)
    print('--------overnight ' + s + ' shape---------')
    print(all_logic_ids.shape)
    all_q_tokens=np.asarray(all_q_tokens)
    return all_q_tokens, all_logic_ids

def load_data_one(maxlen=30, filename=''):
    vocab_dict,_,_,_=load_vocab_all()
    with open(filename, 'r') as f:
        qs = f.readlines()

    res = []
    for q_sentence in qs:
        q_sentence = q_sentence.lower()
        q_sentence = q_sentence.strip('\n')
        token_ids = [_GO]
        token_ids.extend([vocab_dict[x] for x in q_sentence.split()])
        if maxlen>len(token_ids):
            token_ids.extend([ _PAD for i in range(len(token_ids),maxlen)])
        else:
            token_ids = token_ids[:maxlen]
        res.append(token_ids)
    return res

def load_data_sent(maxlen=30, sent=''):
    vocab_dict,_,_,_=load_vocab_all()
	res = []
	q_sentence = sent
	q_sentence = q_sentence.lower()
	q_sentence = q_sentence.strip('\n')
	token_ids = [_GO]
	token_ids.extend([vocab_dict[x] for x in q_sentence.split()])
	if maxlen>len(token_ids):
		token_ids.extend([ _PAD for i in range(len(token_ids),maxlen)])
	else:
		token_ids = token_ids[:maxlen]
	res.append(token_ids)
    return res

def load_data(maxlen=30, load=False, s='train'):
    if s == 'test' or s == 'train' or s == 'dev':
        X, y = load_data_wiki(maxlen=maxlen,load=load,s=s)
        return X, y
    elif s=='overnight':
        X_all, y_all = None, None
        for subset in ['basketball','calendar',...]:
            print('======='+str(subset)+'=======')
            X1, y1 = load_data_overnight(maxlen=maxlen, subset=subset, load=load, s='train')
            X2, y2 = load_data_overnight(maxlen=maxlen, subset=subset, load=load, s='test')
            X = np.concatenate([X1,X2], axis=0)
            if X_all is not None:
                X_all = np.concatenate([X_all, X], axis=0)
            else:
                X_all = X

            y = np.concatenate([y1,y2], axis=0)
            if y_all is not None:
                y_all = np.concatenate([y_all,y], axis=0)
            else:
                y_all = y
        X, y = X_all, y_all
        print('========data '+s+' shape=======')
        print(X.shape)
        print(y.shape)  
        return X, y
    else:
        lists = []
        for subset in ['basketball', 'calendar', ...]:
            X1, y1 = load_data_overnight(maxlen=maxlen, subset=subset, load=load, s='train')
            X2, y2 = load_data_overnight(maxlen=maxlen, subset=subset, load=load, s='test')
            X = np.concatenate([X1,X2], axis=0)
            y = np.concatenate([y1,y2], axis=0)
            lists.append((X, y))        

        return lists


def rebuild(args):
    #build_vocab_all(load=False)
    #load_vocab_all(load=False)

    maxlen = args.maxlen
    load_data(maxlen=maxlen, load=False, s='train')
    load_data(maxlen=maxlen, load=False, s='test')
    load_data(maxlen=maxlen, load=False, s='dev')
    #load_data(maxlen=maxlen, load=False, s='overnight')

if __name__ == "__main__":
    rebuild = True   
    if rebuild:
        #build_vocab_all(load=False)
        #load_vocab_all(load=False)
        build_vocab_all(load=True)
        load_vocab_all(load=True)
        maxlen = 60
        load_data(maxlen=maxlen, load=False, s='train')
        load_data(maxlen=maxlen, load=False, s='test')
        load_data(maxlen=maxlen, load=False, s='dev')
        #load_data(maxlen=maxlen, load=False, s='overnight')


