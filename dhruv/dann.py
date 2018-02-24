from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle as pkl
from sklearn.datasets import make_moons, make_blobs
from sklearn.decomposition import PCA

from flip_gradient import flip_gradient
from utils import *

batch_size = 16

def build_model(shallow_domain_classifier=True):
    X = tf.placeholder(tf.float32, [None, 2], name='X') # Input data
    Y_ind = tf.placeholder(tf.int32, [None], name='Y_ind')  # Class index
    D_ind = tf.placeholder(tf.int32, [None], name='D_ind')  # Domain index
    train = tf.placeholder(tf.bool, [], name='train')       # Switch for routing data to class predictor
    l = tf.placeholder(tf.float32, [], name='l')        # Gradient reversal scaler

    Y = tf.one_hot(Y_ind, 2)
    D = tf.one_hot(D_ind, 2)

    # Feature extractor - single layer
    W0 = weight_variable([2, 15])
    b0 = bias_variable([15])
    F = tf.nn.relu(tf.matmul(X, W0) + b0, name='feature')

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


build_model()    