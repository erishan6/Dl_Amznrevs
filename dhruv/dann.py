from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA



import tensorflow as tf
from tensorflow.contrib import learn
from read import loadDataForDANN
import numpy as np

from flip_gradient import flip_gradient
from utils import *

batch_size = 16


def data(domains):
    x_text, y = loadDataForDANN(domains, "train")
    # max_document_length = max([len(x.split(" ")) for x in x_text])
    max_document_length = 200
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    return x, y, len(vocab_processor.vocabulary_)


def build_model(sequence_length, vocab_size, shallow_domain_classifier=True):
    embedding_size = 128
    X = tf.placeholder(tf.int32, [None, sequence_length], name='X')  # Input data

    # X = tf.placeholder(tf.int32, [None, sequence_length], name="X")
    Y_ind = tf.placeholder(tf.int32, [None], name='Y_ind')  # Class index
    D_ind = tf.placeholder(tf.int32, [None], name='D_ind')  # Domain index
    train = tf.placeholder(tf.bool, [], name='train')       # Switch for routing data to class predictor
    l = tf.placeholder(tf.float32, [], name='l')        # Gradient reversal scaler

    Y = tf.one_hot(Y_ind, 2)
    D = tf.one_hot(D_ind, 2)

    # embedding layer
    W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
    embedded_chars = tf.nn.embedding_lookup(W, X)
    embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
    print(embedded_chars_expanded.shape)
    conv2 = tf.layers.conv2d(
      inputs=embedded_chars_expanded,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])


    # Feature extractor - single layer
    print(pool2_flat.shape[1])
    W0 = weight_variable([int(str(pool2_flat.shape[1])), 15])
    b0 = bias_variable([15])
    F = tf.nn.relu(tf.matmul(pool2_flat, W0) + b0, name='feature')

    # Label predictor - single layer
    f = tf.cond(train, lambda: tf.slice(F, [0, 0], [batch_size // 2, -1]), lambda: F)
    y = tf.cond(train, lambda: tf.slice(Y, [0, 0], [batch_size // 2, -1]), lambda: Y)

    W1 = weight_variable([15, 2])
    b1 = bias_variable([2])
    p_logit = tf.matmul(f, W1) + b1
    p = tf.nn.softmax(p_logit)
    p_loss = tf.nn.softmax_cross_entropy_with_logits(logits=p_logit, labels=y)

    # Domain predictor - shallow
    f_ = flip_gradient(F, l)

    if shallow_domain_classifier:
        W2 = weight_variable([15, 2])
        b2 = bias_variable([2])
        d_logit = tf.matmul(f_, W2) + b2
        d = tf.nn.softmax(d_logit)
        d_loss = tf.nn.softmax_cross_entropy_with_logits(logits=d_logit, labels=D)

    else:
        W2 = weight_variable([15, 8])
        b2 = bias_variable([8])
        h2 = tf.nn.relu(tf.matmul(f_, W2) + b2)

        W3 = weight_variable([8, 2])
        b3 = bias_variable([2])
        d_logit = tf.matmul(h2, W3) + b3
        d = tf.nn.softmax(d_logit)
        d_loss = tf.nn.softmax_cross_entropy_with_logits(logits=d_logit, labels=D)


    # Optimization
    pred_loss = tf.reduce_sum(p_loss, name='pred_loss')
    domain_loss = tf.reduce_sum(d_loss, name='domain_loss')
    total_loss = tf.add(pred_loss, domain_loss, name='total_loss')

    pred_train_op = tf.train.AdamOptimizer().minimize(pred_loss, name='pred_train_op')
    domain_train_op = tf.train.AdamOptimizer().minimize(domain_loss, name='domain_train_op')
    dann_train_op = tf.train.AdamOptimizer().minimize(total_loss, name='dann_train_op')

    # Evaluation
    p_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(p, 1)), tf.float32), name='p_acc')
    d_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(D, 1), tf.argmax(d, 1)), tf.float32), name='d_acc')


def train_and_evaluate(Xs, ys, Xt, yt, sess, train_op_name, train_loss_name, grad_scale=None, num_batches=10000, verbose=True):
    # Create batch builders
    S_batches = batch_generator([Xs, ys], batch_size // 2)
    T_batches = batch_generator([Xt, yt], batch_size // 2)

    # Get output tensors and train op
    d_acc = sess.graph.get_tensor_by_name('d_acc:0')
    p_acc = sess.graph.get_tensor_by_name('p_acc:0')
    train_loss = sess.graph.get_tensor_by_name(train_loss_name + ':0')
    train_op = sess.graph.get_operation_by_name(train_op_name)

    sess.run(tf.global_variables_initializer())
    for i in range(num_batches):

        # If no grad_scale, use a schedule
        if grad_scale is None:
            p = float(i) / num_batches
            lp = 2. / (1. + np.exp(-10. * p)) - 1
        else:
            lp = grad_scale

        X0, y0 = next(S_batches)
        X1, y1 = next(T_batches)
        print(X0[0])
        print(X1[1])
        Xb = np.vstack([X0, X1])
        yb = np.hstack([y0, y1])
        D_labels = np.hstack([np.zeros(batch_size // 2, dtype=np.int32),
                              np.ones(batch_size // 2, dtype=np.int32)])

        _, loss, da, pa = sess.run([train_op, train_loss, d_acc, p_acc],
                                   feed_dict={'X:0': Xb, 'Y_ind:0': yb, 'D_ind:0': D_labels,
                                              'train:0': True, 'l:0': lp})

        if verbose and i % 200 == 0:
            print('loss: {}, domain accuracy: {}, class accuracy: {}'.format(loss, da, pa))


    # Get final accuracies on whole dataset
    das, pas = sess.run([d_acc, p_acc], feed_dict={'X:0': Xs, 'Y_ind:0': ys,
                            'D_ind:0': np.zeros(Xs.shape[0], dtype=np.int32), 'train:0': False, 'l:0': 1.0})
    dat, pat = sess.run([d_acc, p_acc], feed_dict={'X:0': Xt, 'Y_ind:0': yt,
                            'D_ind:0': np.ones(Xt.shape[0], dtype=np.int32), 'train:0': False, 'l:0': 1.0})

    print('Source domain: ', das)
    print('Source class: ', pas)
    print('Target domain: ', dat)
    print('Target class: ', pat)


def extract_and_plot_pca_feats(sess, feat_tensor_name='feature'):
    F = sess.graph.get_tensor_by_name(feat_tensor_name + ':0')
    emb_s = sess.run(F, feed_dict={'X:0': Xs})
    emb_t = sess.run(F, feed_dict={'X:0': Xt})
    emb_all = np.vstack([emb_s, emb_t])

    pca = PCA(n_components=2)
    pca_emb = pca.fit_transform(emb_all)

    num = pca_emb.shape[0] // 2
    plt.scatter(pca_emb[:num,0], pca_emb[:num,1], c=ys, cmap='coolwarm', alpha=0.4)
    plt.scatter(pca_emb[num:,0], pca_emb[num:,1], c=yt, cmap='cool', alpha=0.4)
    plt.show()


if __name__ == '__main__':
    xs, ys, vs1 = data(["music"])
    xt, yt, vs2 = xs, ys, vs1
    print(len(xs))
    print(vs1)
    print(vs2)
    build_model(max(xs.shape[1], xt.shape[1]), max(vs1, vs2))
    sess = tf.InteractiveSession()
    train_and_evaluate(xs, ys, xt, yt, sess, 'pred_train_op', 'pred_loss', verbose=False)
    extract_and_plot_pca_feats(sess)
