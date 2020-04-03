import numpy as np
import os
import pickle
path = os.path.abspath(__file__)
datapath = os.path.dirname(path)
def load_data(file1='wiki.npz', file2='wiki_label.npz', datapath=datapath):
    data = np.load(os.path.join(datapath,file1))
    X_train_qu = data['X_train_qu']
    X_train_col = data['X_train_col']
    X_test_qu = data['X_test_qu']
    X_test_col = data['X_test_col']
    X_dev_qu = data['X_dev_qu']
    X_dev_col = data['X_dev_col']

    data = np.load(os.path.join(datapath,file2))
    y_train = data['y_train'].reshape(-1, 1)
    y_test = data['y_test'].reshape(-1, 1)
    y_dev = data['y_dev'].reshape(-1, 1)

    return X_train_qu, X_train_col, y_train, X_test_qu, X_test_col, y_test, X_dev_qu, X_dev_col, y_dev

def load_data_char(file1='wiki_char.npz', file2='wiki_label.npz'):
    data = np.load(os.path.join(datapath,file1))
    X_train_qu = data['X_train0']
    X_train_col = data['X_train1']
    X_test_qu = data['X_test0']
    X_test_col = data['X_test1']
    X_dev_qu = data['X_dev0']
    X_dev_col = data['X_dev1']

    data = np.load(os.path.join(datapath,file2))
    y_train = data['y_train']
    y_test = data['y_test']
    y_dev = data['y_dev']

    return X_train_qu, X_train_col, y_train, X_test_qu, X_test_col, y_test, X_dev_qu, X_dev_col, y_dev

def load_data_word(file1='wiki_word.npz', file2='wiki_label.npz',datapath=datapath):
    data = np.load(os.path.join(datapath,file1))
    X_train_qu = data['X_train0']
    X_train_col = data['X_train1']
    X_test_qu = data['X_test0']
    X_test_col = data['X_test1']
    X_dev_qu = data['X_dev0']
    X_dev_col = data['X_dev1']

    data = np.load(os.path.join(datapath,file2))
    y_train = data['y_train'].reshape(-1, 1)
    y_test = data['y_test'].reshape(-1, 1)
    y_dev = data['y_dev'].reshape(-1, 1)

    return X_train_qu, X_train_col, y_train, X_test_qu, X_test_col, y_test, X_dev_qu, X_dev_col, y_dev

if __name__ == '__main__':
    #X_train_qu, X_train_col, y_train, X_test_qu, X_test_col, y_test, X_dev_qu, X_dev_col, y_dev = load_data_word()
    X_train_qu, X_train_col, y_train, X_test_qu, X_test_col, y_test, X_dev_qu, X_dev_col, y_dev = load_data()
    datapath = '/home/wzw0022/end2end_anno/data'
    X_train_qu, X_train_col, y_train, X_test_qu1, X_test_col1, y_test, X_dev_qu, X_dev_col, y_dev = load_data(datapath=datapath)
    #X_train_qu, X_train_col, y_train, X_test_qu1, X_test_col1, y_test, X_dev_qu1, X_dev_col1, y_dev = load_data_word(datapath=datapath)
    from embed import glove
    g = glove.Glove()
    for x1, x2 in zip(X_test_qu, X_test_qu1):
        if (x1 != x2).any():
            print('-------')
            print(len(x1))
            print(len(x2))
            print(' '.join([g.reverse_embedding(x) for x in x2]))
            print(' '.join([g.reverse_embedding(x) for x in x1]))

    '''
    vocab_size = 1224
    with open('/home/wzw0022/end2end_anno/data/tokenizer.pickle', 'rb') as handle:
        tkr = pickle.load(handle)
    tkr.word_index[' '] = vocab_size + 1
    tkr.word_index['^'] = vocab_size + 2
    tkr.word_index['$'] = vocab_size + 3    
    
    
    reverse_word_map = dict(map(reversed, tkr.word_index.items())) 
    reverse_word_map[47] = '$'
    for i,(a, b) in enumerate(zip(X_dev_col, X_dev_col1)):
        if not (a==b).all():
            print('============')
            print(' '.join([str(x) for x in a]))
            print(' '.join([reverse_word_map.get(x) for x in a]))
            print(' '.join([str(x) for x in b]))
            print(' '.join([reverse_word_map.get(x) for x in b]))   
    for i,(a, b) in enumerate(zip(X_dev_qu, X_dev_qu1)):
        if not (a==b).all():
            pass
            #print('============')
            #print(' '.join([str(x) for x in a]))
            #print(' '.join([reverse_word_map.get(x) for x in a]))
            #print(' '.join([str(x) for x in b]))
            #print(' '.join([reverse_word_map.get(x) for x in b]))
    '''
