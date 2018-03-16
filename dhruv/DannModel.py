# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import pickle as pkl
# from sklearn.manifold import TSNE

from flip_gradient import flip_gradient
from utils import *
import time
import os
import pickle
import itertools
import sys

import logging
logging.basicConfig(filename='train.log',level=logging.DEBUG)

batch_size = 64
max_document_length = 200
sequence_length = 200
default_num_steps = 201
embedding_size = 128
log_frequency = 50
max_models_to_keep = 1

# static strings
train_pickle_file_suffix = ".train.pickle"
BASE_DATA_DIR = "./SentimentClassification/"


class DannModel(object):
    """Simple MNIST domain adaptation model."""
    def __init__(self, vocab_size):
        self._build_model(vocab_size)

    def _build_model(self, vocab_size):
        # pixel_mean = np.vstack([loadDataForDANN(["books"], "train"), loadDataForDANN(["music"], "train")]).mean((0,1,2))
        # X should be in the form of embedding vector
        # this is the shape after embedding layer
        # self.X = tf.placeholder(tf.uint8, [None, 28, 28, 1])
        self.X = tf.placeholder(tf.int32, [None, sequence_length], name="X")
        self.y = tf.placeholder(tf.float32, [None, 2], name="y")
        self.domain = tf.placeholder(tf.float32, [None, 2], name="domain")
        self.l = tf.placeholder(tf.float32, [], name="l")
        self.train = tf.placeholder(tf.bool, [], name="train")

        # X_input = (tf.cast(self.X, tf.float32))# - pixel_mean) / 255.

        #

        # X = tf.placeholder(tf.int32, [None, sequence_length], name='X')  # Input data

        # X = tf.placeholder(tf.int32, [None, sequence_length], name="X")
        # Y_ind = tf.placeholder(tf.int32, [None], name='Y_ind')  # Class index
        # D_ind = tf.placeholder(tf.int32, [None], name='D_ind')  # Domain index
        # train = tf.placeholder(tf.bool, [], name='train')       # Switch for routing data to class predictor
        # l = tf.placeholder(tf.float32, [], name='l')        # Gradient reversal scaler

        # Y = tf.one_hot(Y_ind, 2)
        # D = tf.one_hot(D_ind, 2)

        # embedding layer
        W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
        embedded_chars = tf.nn.embedding_lookup(W, self.X)
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
        print(embedded_chars_expanded.shape)

        #

        # CNN model for feature extraction
        with tf.name_scope('feature_extractor'):

            W_conv0 = weight_variable([5, 5, 1, 32])
            b_conv0 = bias_variable([32])
            h_conv0 = tf.nn.relu(conv2d(embedded_chars_expanded, W_conv0) + b_conv0)
            h_pool0 = max_pool_2x2(h_conv0)

            W_conv1 = weight_variable([5, 5, 32, 48])
            b_conv1 = bias_variable([48])
            h_conv1 = tf.nn.relu(conv2d(h_pool0, W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)

            # The domain-invariant feature
            self.feature = tf.reshape(h_pool1, [-1, 40 * 40 * 48])

        # MLP for class prediction
        with tf.name_scope('label_predictor'):

            # Switches to route target examples (second half of batch) differently
            # depending on train or test mode.
            all_features = lambda: self.feature
            source_features = lambda: tf.slice(self.feature, [0, 0], [batch_size // 2, -1])
            classify_feats = tf.cond(self.train, source_features, all_features)

            all_labels = lambda: self.y
            source_labels = lambda: tf.slice(self.y, [0, 0], [batch_size // 2, -1])
            self.classify_labels = tf.cond(self.train, source_labels, all_labels)

            W_fc0 = weight_variable([40 * 40 * 48, 100])
            b_fc0 = bias_variable([100])
            h_fc0 = tf.nn.relu(tf.matmul(classify_feats, W_fc0) + b_fc0)

            W_fc1 = weight_variable([100, 100])
            b_fc1 = bias_variable([100])
            h_fc1 = tf.nn.relu(tf.matmul(h_fc0, W_fc1) + b_fc1)

            W_fc2 = weight_variable([100, 2])
            b_fc2 = bias_variable([2])
            logits = tf.matmul(h_fc1, W_fc2) + b_fc2

            self.pred = tf.nn.softmax(logits, name="prediction")
            self.pred_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.classify_labels)

        # Small MLP for domain prediction with adversarial loss
        with tf.name_scope('domain_predictor'):

            # Flip the gradient when backpropagating through this operation
            feat = flip_gradient(self.feature, self.l)

            d_W_fc0 = weight_variable([40 * 40 * 48, 100])
            d_b_fc0 = bias_variable([100])
            d_h_fc0 = tf.nn.relu(tf.matmul(feat, d_W_fc0) + d_b_fc0)

            d_W_fc1 = weight_variable([100, 2])
            d_b_fc1 = bias_variable([2])
            d_logits = tf.matmul(d_h_fc0, d_W_fc1) + d_b_fc1

            self.domain_pred = tf.nn.softmax(d_logits)
            self.domain_loss = tf.nn.softmax_cross_entropy_with_logits(logits=d_logits, labels=self.domain)

def training(d1, d2, d3, TAG):
    def train_and_evaluate(training_mode, graph, model, xs, ys, xt,yt, xtest,ytest, TAG, num_steps=default_num_steps, verbose=True):
        """Helper to run the model with different training modes."""

        with tf.Session(graph=graph) as sess:
            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", TAG + "-" + timestamp))
            logging.info("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            # pred_loss_summary = tf.summary.scalar("pred_loss", model.pred_loss)
            # domain_loss_summary = tf.summary.scalar("domain_loss", model.domain_loss)
            # total_loss_summary = pred_loss_summary + domain_loss_summary
            domain_acc_summary = tf.summary.scalar("domain_accuracy", domain_acc)
            label_acc_summary = tf.summary.scalar("label_accuracy", label_acc)

            # Train Summaries
            train_summary_op = tf.summary.merge([domain_acc_summary, label_acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # # Dev summaries
            # dev_summary_op = tf.summary.merge([pred_loss_summary, domain_acc_summary, domain_acc_summary, label_acc_summary])
            # dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            # dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_models_to_keep)
            tf.global_variables_initializer().run()

            # Batch generators
            gen_source_batch = batch_generator([xs, ys], batch_size // 2)
            gen_target_batch = batch_generator([xt, yt], batch_size // 2)
            gen_source_only_batch = batch_generator([xs, ys], batch_size)
            gen_target_only_batch = batch_generator([xt, yt], batch_size)

            domain_labels = np.vstack([np.tile([1., 0.], [batch_size // 2, 1]),
                                       np.tile([0., 1.], [batch_size // 2, 1])])

            # Training loop
            for i in range(num_steps):

                # Adaptation param and learning rate schedule as described in the paper
                p = float(i) / num_steps
                l = 2. / (1. + np.exp(-10. * p)) - 1
                lr = 0.01 / (1. + 10 * p)**0.75

                # Training step
                if training_mode == 'dann':

                    X0, y0 = next(gen_source_batch)
                    X1, y1 = next(gen_target_batch)
                    X = np.vstack([X0, X1])
                    y = np.vstack([y0, y1])

                    _, batch_loss, summaries, dloss, ploss, d_acc, p_acc = sess.run(
                        [dann_train_op, total_loss, train_summary_op, domain_loss, pred_loss, domain_acc, label_acc],
                        feed_dict={model.X: X, model.y: y, model.domain: domain_labels,
                                   model.train: True, model.l: l, learning_rate: lr})
                    train_summary_writer.add_summary(summaries, i)

                    if verbose and i % log_frequency == 0:
                        logging.info('loss: {}  d_acc: {}  p_acc: {}  p: {}  l: {}  lr: {}'.format(
                                batch_loss, d_acc, p_acc, p, l, lr))
                        path = saver.save(sess, checkpoint_prefix, global_step=i)
                        logging.info("Saved model checkpoint to {}\n".format(path))

                elif training_mode == 'source':
                    X, y = next(gen_source_only_batch)
                    _, batch_loss = sess.run([regular_train_op, pred_loss],
                                         feed_dict={model.X: X, model.y: y, model.train: False,
                                                    model.l: l, learning_rate: lr})

                elif training_mode == 'target':
                    X, y = next(gen_target_only_batch)
                    _, batch_loss = sess.run([regular_train_op, pred_loss],
                                         feed_dict={model.X: X, model.y: y, model.train: False,
                                                    model.l: l, learning_rate: lr})

        #     gen_test_batch = batch_generator([xtest, ytest], batch_size)
        #     X0, y0 = next(gen_test_batch)
        #     X = np.vstack([X0])
        #     y = np.vstack([y0])
        #     # Compute final evaluation on test data
        #     source_acc = sess.run(label_acc,
        #                         feed_dict={model.X: X, model.y: y, model.domain: domain_labels, model.train: False})

        #     target_acc = sess.run(label_acc,
        #                         feed_dict={model.X: X, model.y: y, model.domain: domain_labels, model.train: False})

        #     test_domain_acc = sess.run(domain_acc,
        #                         feed_dict={model.X: X, model.y: y, model.domain: domain_labels, model.l: 1.0, model.train: False})

        #     test_emb = sess.run(model.feature, feed_dict={model.X: X, model.train: False})
        # return source_acc, target_acc, test_domain_acc, test_emb

    # Build the model graph
    graph = tf.get_default_graph()
    xs, ys, vs1 = data([d1], "train", train_pickle_file_suffix)
    xt,yt,vs2 = data([d2], "train", train_pickle_file_suffix)
    xtest,ytest,vs3 = data([d3], "train", train_pickle_file_suffix)

    with graph.as_default():
        model = DannModel(max(vs1,vs2))

        learning_rate = tf.placeholder(tf.float32, [])

        pred_loss = tf.reduce_mean(model.pred_loss)
        domain_loss = tf.reduce_mean(model.domain_loss)
        total_loss = pred_loss + domain_loss

        regular_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(pred_loss)
        dann_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_loss)

        # Evaluation
        correct_label_pred = tf.equal(tf.argmax(model.classify_labels, 1), tf.argmax(model.pred, 1))
        label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))
        correct_domain_pred = tf.equal(tf.argmax(model.domain, 1), tf.argmax(model.domain_pred, 1))
        domain_acc = tf.reduce_mean(tf.cast(correct_domain_pred, tf.float32))

    print('\nDomain adaptation training')
    # source_acc, target_acc, test_domain_acc, dann_emb =
    train_and_evaluate('dann', graph, model, xs, ys,xt,yt,xtest,ytest, TAG)
    # print('Source accuracy:', source_acc)
    # print('Target accuracy:', target_acc)
    # print('Test domain accuracy:', test_domain_acc)

if __name__ == '__main__':
        tag = sys.argv[1] + "-" + sys.argv[2]
        logging.info("#################### training for " + tag + "####################")
        training(sys.argv[2], sys.argv[2], sys.argv[3], tag)
        logging.info("#################### ending for " + tag + "####################")
