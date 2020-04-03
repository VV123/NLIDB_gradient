import numpy as np
import os
path = os.path.abspath(__file__)
datapath = os.path.dirname(path).replace('utils/end2end_anno/data/embed','data/end2end')
savepath = os.path.dirname(path).replace('/embed','')
import codecs
import nltk
from embed import glove

def embed_data(maxlen_p=60,maxlen_q=3,embedding=None,save=False):
	filepath ='wiki.npz'
	filepath_X = os.path.expanduser(os.path.join(savepath, filepath))
	filepath = 'wiki_label.npz'
	filepath_y = os.path.expanduser(os.path.join(savepath, filepath))
	g = embedding
	if g is None:
		g = glove.Glove()
	
	import nltk
	def _embedding(fpath):
		for line in codecs.open(fpath,'r','utf-8-sig'):
			assert len(line.split('\t')) == 3 or line.startswith('#')
		questions = [ nltk.word_tokenize(line.split('\t')[0]) for line in codecs.open(fpath,'r','utf-8-sig') if not line.startswith('#')]
		cols = [ nltk.word_tokenize(line.split('\t')[1]) for line in codecs.open(fpath,'r','utf-8-sig') if not line.startswith('#')]
		labels = [ line.split('\t')[2] for line in codecs.open(fpath,'r','utf-8-sig') if not line.startswith('#')]

		return g.embedding(questions, maxlen=maxlen_p-1), g.embedding(cols, maxlen=maxlen_q-1), 

	def _read_label(fpath):
		labels = [ line.split('\t')[2] for line in codecs.open(fpath,'r','utf-8-sig') if not line.startswith('#')]
		return labels

	print('\nGenerating training/test data')
	X_train_p,X_train_q = _embedding(os.path.join(datapath, 'train_model.txt'))
	X_test_p,X_test_q = _embedding(os.path.join(datapath, 'test_model.txt'))
	X_dev_p,X_dev_q = _embedding(os.path.join(datapath, 'dev_model.txt'))
	X_train_ans = _read_label(os.path.join(datapath, 'train_model.txt'))
	X_test_ans = _read_label(os.path.join(datapath, 'test_model.txt'))
	X_dev_ans = _read_label(os.path.join(datapath, 'dev_model.txt'))

	if save:
		print('\nSaving')
		np.savez(filepath_y, y_train=X_train_ans, y_test=X_test_ans, y_dev=X_dev_ans)
		np.savez(filepath_X, X_train_qu=X_train_p, X_train_col=X_train_q, X_test_qu=X_test_p, X_test_col=X_test_q, X_dev_qu=X_dev_p, X_dev_col=X_dev_q)
		print('\nSaved!')

if __name__ == "__main__":
	embed_data()
