from __future__ import print_function
import sys
import os
import keras
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from utils.data_manager import load_vocab_all
from collections import defaultdict
import argparse
from argparse import ArgumentParser
from decode_helper import decode_data_recover, decode_data_recover_overnight, decode_data
from model import construct_graph
import sys
reload(sys)
sys.setdefaultencoding('utf8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# ----------------------------------------------------------------------------
_PAD = 0
_GO = 1
_END = 2
# ----------------------------------------------------------------------------
def train(sess, env, X_data, y_data, epochs=10, load=False, shuffle=True, batch_size=128,
          name='model', base=0, model2Bload=''):
    """
    Train TF model by env.train_op
    """
    if load:
        print('\nLoading saved model')
        env.saver.restore(sess, model2Bload )

    print('\nTrain model')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    for epoch in range(epochs):
        print('\nEpoch {0}/{1}'.format(epoch+1, epochs))
        sys.stdout.flush()
        if shuffle:
            print('\nShuffling data')
            ind = np.arange(n_sample)
            np.random.shuffle(ind)
            X_data = X_data[ind]
            y_data = y_data[ind]

        for batch in range(n_batch):
            print(' batch {0}/{1}'.format(batch+1, n_batch),end='\r')
            start = batch * batch_size
            end = min(n_sample, start+batch_size)
            sess.run(env.train_op, feed_dict={env.x: X_data[start:end],
                                              env.y: y_data[start:end],
                                              env.training: True})
        
        evaluate(sess, env, X_data, y_data, batch_size=batch_size)

        if (epoch+1) == epochs:
            print('\n Saving model')
            env.saver.save(sess, 'model/{0}-{1}'.format(name, base))
    return 'model/{0}-{1}'.format(name, base) 

def evaluate(sess, env, X_data, y_data, batch_size=128):
    """
    Evaluate TF model by running env.loss and env.acc.
    """
    print('\nEvaluating')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    loss, acc = 0, 0

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch+1, n_batch),end='\r')
        sys.stdout.flush()
        start = batch * batch_size
        end = min(n_sample, start+batch_size)
        cnt = end - start
        batch_loss, batch_acc = sess.run(
            [env.loss,env.acc],
            feed_dict={env.x: X_data[start:end],
                       env.y: y_data[start:end]})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
    loss /= n_sample
    acc /= n_sample

    print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return acc

#---------------------------------------------------------------------------
