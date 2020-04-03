import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from data import load_data, load_data_word
import numpy as np
import tensorflow as tf
from numpy import linalg as LA
from plot import plot
base = os.path.dirname(os.path.abspath(__file__))
# ----------------------------------------------------------------------------
maxlen0 = 60
maxlen1 = 3
embedding_dim = 300
batch_size = 16
train_batch_size = 128
n_states = 300
classes = 2
train_needed = False
train_epochs = 10
save_gradient = True
# ----------------------------------------------------------------------------
maxwlen = 10
cmaxlen0 = maxlen0  * (maxwlen
                     + 2         # start/end of word symbol
                     + 1)        # whitespace between tokens
cmaxlen1 = maxlen1 * (maxwlen
                     + 2         # start/end of word symbol
                     + 1)        # whitespace between tokens
# ---------------------------------------------------------------------------
def matcher(text, key):
    """Match LSTM.

    The match LSTM could be thought as encoding text according to the
    attention weight, w.r.t key.

    :param text: [B, T0, D]
    :param key: [B, T1, D]

    :returns: [B, T1, D]
    """
    key_shape = tf.shape(key)
    print('key shape:')
    print(key.get_shape().as_list())
    B, T1 = key_shape[0], key_shape[1]

    const_key_shape = key.get_shape().as_list()
    D = const_key_shape[-1]

    cell = tf.nn.rnn_cell.LSTMCell(D)

    # [B, T1, D] => [T1, B, D]
    key = tf.transpose(key, [1, 0, 2])

    w0 = tf.get_variable('w0', [D, D], tf.float32)
    w1 = tf.get_variable('w1', [D, D], tf.float32)
    w2 = tf.get_variable('w2', [D, D], tf.float32)
    v0 = tf.get_variable('v0', [1, D], tf.float32)
    v1 = tf.get_variable('v1', [D, 1], tf.float32)
    s0 = tf.get_variable('s0', (), tf.float32)

    a = tf.tensordot(text, w0, [[2], [0]])
    b = tf.tensordot(key, w1, [[2], [0]])

    def _cond(i, *_):
        return tf.less(i, T1)

    def _body(i, outputs, state):
        b_i = tf.expand_dims(tf.gather(b, i), axis=1)
        key_i = tf.gather(key, i)

        # [B, 1, D]
        c = tf.expand_dims(tf.matmul(state[1], w2) + v0, axis=1)
        # [B, T0, 1]
        d = tf.tensordot(tf.tanh(a + b_i + c), v1, [[2], [0]]) + s0
        # [B, 1, T0]
        attn = tf.nn.softmax(tf.transpose(d, [0, 2, 1]))
        # [B, D]
        ciphertext = tf.squeeze(tf.matmul(attn, text), axis=1)
        # [B, 2D]
        e = tf.concat((key_i, ciphertext), axis=1)

        output, new_state = cell(e, state)
        outputs = outputs.write(i, output)

        return i + 1, outputs, new_state

    outputs = tf.TensorArray(tf.float32, T1)
    state = cell.zero_state(B, tf.float32)
    _, outputs, _ = tf.while_loop(_cond, _body, (0, outputs, state),
                                  name='matcher')

    # [T1, B, D] => [B, T1, D]
    outputs = tf.transpose(outputs.stack(), [1, 0, 2])
    outputs.set_shape(const_key_shape)

    return outputs



# ------------------------------------------------------------------------------
def train(sess, env, cenv, X0_data, X1_data, CX0_data, CX1_data, y_data, X0_valid, X1_valid, CX0_valid, CX1_valid, y_valid ,epochs=1, load=False,
          shuffle=True, batch_size=128, name='model'):
    """
    Train a TF model by running env.train_op.
    """
    if load:
        print('\nLoading saved model')
        return env.saver.restore(sess, os.path.join(base, 'model/{}'.format(name)))

    print('\nTrain model')
    n_sample = X0_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    for epoch in range(epochs):
        print('\nEpoch {0}/{1}'.format(epoch+1, epochs))

        if shuffle:
            print('\nShuffling data')
            ind = np.arange(n_sample)
            np.random.shuffle(ind)
            X0_data = X0_data[ind]
            X1_data = X1_data[ind]
            CX0_data = CX0_data[ind]
            CX1_data = CX1_data[ind]
            y_data = y_data[ind]

        for batch in range(n_batch):
            print(' batch {0}/{1}'.format(batch+1, n_batch), end='\r')
            start = batch * batch_size
            end = min(n_sample, start+batch_size)
            cnt = end - start
            if cnt < batch_size:
                break
            sess.run(env.train_op, feed_dict={env.x0: X0_data[start:end],
                                              env.x1: X1_data[start:end],
                                              cenv.X0: CX0_data[start:end],
                                              cenv.X1: CX1_data[start:end],
                                              env.y: y_data[start:end]})
        evaluate(sess, env, cenv, X0_valid, X1_valid, CX0_valid, CX1_valid, y_valid, batch_size=batch_size)

    print('\n Saving model')
    env.saver.save(sess, os.path.join(base, 'model/{}'.format(name)))

def evaluate(sess, env, cenv, X0_data, X1_data, CX0_data, CX1_data, y_data, batch_size=128):
    """
    Evaluate TF model by running env.loss and env.acc.
    """
    print('\nEvaluating')
    n_sample = X0_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    loss, acc = 0, 0
    dy_dx, dy_dx_char = [], []
    ybar = []
    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch+1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start+batch_size)
        cnt = end - start
        if cnt < batch_size:
            break
        batch_loss, batch_acc, batch_dy_dx, batch_dy_dx_char, batch_ybar = sess.run(
            [env.loss, env.acc, env.dy_dx, cenv.dy_dx_char, env.ybar],
            feed_dict={env.x0: X0_data[start:end],
                       env.x1: X1_data[start:end],
                       cenv.X0: CX0_data[start:end],
                       cenv.X1: CX1_data[start:end],
                       env.y: y_data[start:end]})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
        wnorms = LA.norm(batch_dy_dx, axis=2)
        cnorms = LA.norm(batch_dy_dx_char, axis=2)
        dy_dx_char = np.vstack([dy_dx_char, cnorms]) if dy_dx_char != [] else cnorms
        dy_dx = np.vstack([dy_dx, wnorms]) if dy_dx != [] else wnorms
        ybar.extend(batch_ybar)
    loss /= n_sample
    acc /= n_sample

    print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return loss, acc, dy_dx, dy_dx_char, ybar

# ------------------------------------------------------------------------------

class Dummy:
    pass

env = Dummy()
cenv = Dummy()

def build_model(dataset='test', env=env, cenv=cenv, batch_size=None):
    
    if batch_size == None:
        if dataset == 'test':
            batch_size = 16
        elif dataset == 'train':
            batch_size = 8
        elif dataset == 'dev':
            batch_size = 207

    env.x0 = tf.placeholder(tf.float32, (batch_size, maxlen0, embedding_dim),
                            name='x0')
    env.x1 = tf.placeholder(tf.float32, (batch_size, maxlen1, embedding_dim),
                            name='x1')
    env.y = tf.placeholder(tf.float32, (batch_size, 1), name='y')
    #print('Input-word shape:')
    #print(env.x0.get_shape().as_list())
    #print(env.x1.get_shape().as_list())
    # Convert to internal representation
    #------------------------------------------------------------------------------------------
    cenv.X0 = tf.placeholder(tf.int32, (batch_size, cmaxlen0),
                            name='char_x0')
    cenv.X1 = tf.placeholder(tf.int32, (batch_size, cmaxlen1),
                            name='char_x1')

    embedding = tf.get_variable('embedding', [1224+3+1, embedding_dim], dtype=tf.float32)
    cenv.x0 = tf.nn.embedding_lookup(embedding, cenv.X0)
    cenv.x1 = tf.nn.embedding_lookup(embedding, cenv.X1)
    # Convert to internal representation
    B = cenv.X0.get_shape().as_list()[0]
    cenv.x0 = tf.reshape(cenv.x0, [B * maxlen0, maxwlen+3, embedding_dim]) #[-1, maxlen0, maxwlen+3]
    cenv.x1 = tf.reshape(cenv.x1, [B * maxlen1, maxwlen+3, embedding_dim]) #[-1, maxlen1, maxwlen+3]
    #print('Input-char shape:')
    #print(cenv.x0.get_shape().as_list())
    #print(cenv.x1.get_shape().as_list())

    zs0 = [tf.layers.conv1d(cenv.x0, filters, kernel_size, use_bias=True, activation=tf.tanh) for filters, kernel_size in zip([50, 50, 50, 50, 50, 50], [1, 2, 3, 4, 5, 6])]

    zs1 = [tf.layers.conv1d(cenv.x1, filters, kernel_size, use_bias=True, activation=tf.tanh)    for filters, kernel_size in zip([50, 50, 50, 50, 50, 50], [1, 2, 3, 4, 5, 6])]

    zs0 = [tf.reduce_max(z, axis=1) for z in zs0]
    zs1 = [tf.reduce_max(z, axis=1) for z in zs1]

    z0 = tf.concat(zs0, axis=-1)
    z0 = tf.reshape(z0, [B, maxlen0, -1])

    z1 = tf.concat(zs1, axis=-1)
    z1 = tf.reshape(z1, [B, maxlen1, -1])

    #print('char level shape before concat:')
    #print(z0.get_shape().as_list())
    #print(z1.get_shape().as_list())

    #-------------------concat-----------------------
    x0_concat = tf.concat((env.x0, z0), axis=2, name='concat-x0')
    x1_concat = tf.concat((env.x1, z1), axis=2, name='concat-x1')
    #------------------------------------------------------------------------

    cell0 = tf.nn.rnn_cell.LSTMCell(n_states)
    H0, _ = tf.nn.dynamic_rnn(cell0, x0_concat, dtype=tf.float32, scope='h0')

    cell1 = tf.nn.rnn_cell.LSTMCell(n_states)
    H1, _ = tf.nn.dynamic_rnn(cell1, x1_concat, dtype=tf.float32, scope='h1')

    #print('H1 shape')
    #print(H1.get_shape().as_list())
    with tf.variable_scope('fw') as scope:
        outputs_fw = matcher(H0, H1)

    with tf.variable_scope('bw') as scope:
        outputs_bw = matcher(H0, tf.reverse(H1, axis=[1]))

    outputs = tf.concat((outputs_fw, outputs_bw), axis=-1, name='h0h1')
    output = tf.reduce_mean(outputs, axis=1)

    #---------------------------------------------------
    logits = tf.layers.dense(output, 1)
    env.ybar = tf.sigmoid(logits)

    with tf.variable_scope('loss'):
        xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=env.y,logits=logits)
        env.loss = tf.reduce_mean(xent)

    with tf.variable_scope('acc'):
        t0 = tf.greater(env.ybar, 0.5)
        t1 = tf.greater(env.y, 0.5)
        count = tf.equal(t0, t1)
        env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

    optimizer = tf.train.AdamOptimizer()
    env.train_op = optimizer.minimize(env.loss)

    cenv.dy_dx_char, = tf.gradients(env.loss, z0)
    env.dy_dx, = tf.gradients(env.loss, env.x0)
    env.saver = tf.train.Saver()
    #print('char level gradient')
    #print(cenv.dy_dx_char.get_shape().as_list())
# ---------------------------------------------------------------------------------
def save_gradient(train_needed=False):

    print('Load data start...')
    X_train_qu, X_train_col, y_train, X_test_qu, X_test_col, y_test, X_dev_qu, X_dev_col, y_dev = load_data()
    CX_train_qu, CX_train_col, _, CX_test_qu, CX_test_col, _, CX_dev_qu, CX_dev_col, _ = load_data_word()
    print('Load data done...')
 

    if train_needed:
        build_model('train', batch_size=train_batch_size)
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        train(sess, env, cenv, X_train_qu, X_train_col, CX_train_qu, CX_train_col, y_train, X_dev_qu, X_dev_col, CX_dev_qu, CX_dev_col, y_dev, epochs=train_epochs, load=False,shuffle=True, batch_size=train_batch_size, name='wc_conv1_model')


    for dataset in ['test', 'dev']:
        print('-----%s------'%dataset)
        tf.reset_default_graph()
        build_model(dataset)
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        env.saver.restore(sess, os.path.join(base, "model/wc_conv1_model"))
        if dataset == 'test':
            batch_size = 16
            _, _, wnorms, cnorms, ybar = evaluate(sess, env, cenv, X_test_qu, X_test_col, CX_test_qu, CX_test_col, y_test, batch_size=batch_size)
        if dataset == 'train':
            batch_size = 8
            _, _, wnorms, cnorms, ybar = evaluate(sess, env, cenv, X_train_qu, X_train_col, CX_train_qu, CX_train_col, y_train, batch_size=batch_size)
        if dataset == 'dev':
            batch_size = 207
            _, _, wnorms, cnorms, ybar = evaluate(sess, env, cenv, X_dev_qu, X_dev_col, CX_dev_qu, CX_dev_col, y_dev, batch_size=batch_size)
        print('\nSaving...')
        np.savez('/nfs_shares/wzw0022_home/%s_gradient_norm_conv1.npz'%dataset, wnorms=wnorms, cnorms=cnorms, ybar=ybar)
        print('\nPickle saved!')		

if __name__ == '__main__':
	if save_gradient:
		save_gradient()
