#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import read
from network import Network
from tensorflow.contrib import learn
from random import random

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 1, "Number of checkpoints to store (default: 1)")
tf.flags.DEFINE_string("source_data", "books", "Source data for training (default: books)")
tf.flags.DEFINE_string("target_data", "music", "Target data for training (default: music)")
tf.flags.DEFINE_float("domain_loss_factor_propagation", 0.1, "domain_loss_factor_propagation for training the loss_domain")
tf.flags.DEFINE_float("domain_train_frequency", -1, "domain training frequency for training the loss_domain. A negative value implies seperate training is switched off.")

tf.flags.DEFINE_boolean("use_adam", True, "Select optimizer to use. Default is AdamOptimizer, else use RMSPropOptimizer")

tf.flags.DEFINE_string("activation_function", "relu", "Select activation function to use. Default is relu")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

print("use_adam=" + str(FLAGS.use_adam))
print("activation_function=" + str(FLAGS.activation_function))
print("domain_loss_factor_propagation=" + str(FLAGS.domain_loss_factor_propagation))
print("domain_train_frequency=" + str(FLAGS.domain_train_frequency))
print("source_data=" + str(FLAGS.source_data))
print("target_data=" + str(FLAGS.target_data))

# Data Preparation
# ==================================================

# Load data
print("Loading data...")
x_text, y = read.loadDataForCNN([FLAGS.source_data,FLAGS.target_data], "train")
y1 = [[1,0] for x in range(len(y)//2)]
y2 = [[0,1] for x in range(len(y)//2)]
y_domain = y1 + y2
# print(y)
y = np.array(y)
y_domain = np.array(y_domain)
# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))
print(x[1])

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]
y_domain_shuffled = y_domain[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
y_domain_train, y_domain_dev = y_domain_shuffled[:dev_sample_index], y_domain_shuffled[dev_sample_index:]

del x, y, x_shuffled, y_shuffled, y_domain, y_domain_shuffled

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = Network(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            activation_function=FLAGS.activation_function,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        if FLAGS.use_adam:
            optimizer = tf.train.AdamOptimizer(1e-3)
        else:
            optimizer = tf.train.RMSPropOptimizer(1e-3)
        #grads_and_vars = optimizer.compute_gradients(cnn.loss + cnn.loss_domain)
        #train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        if FLAGS.domain_train_frequency < 0:
            # seperate training is switched off, use constant equation
            loss_equation = (cnn.loss + (FLAGS.domain_loss_factor_propagation)*(1/cnn.loss_domain))

        elif random()> FLAGS.domain_train_frequency:
            loss_equation = cnn.loss
        else:
            loss_equation = 1/cnn.loss_domain


        grads_and_vars = optimizer.compute_gradients(loss_equation)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        #l_op = optimizer.minimize(cnn.loss)
        #d_op = optimizer.minimize(-1*cnn.loss_domain)
        #grads_and_vars2=optimizer.compute_gradients(-1*cnn.loss_domain)
        #train_op2 = optimizer.apply_gradients(grads_and_vars2, global_step=global_step)
        #train_op = tf.group(train_op1, train_op2)
        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", loss_equation)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch, y_domain_batch):
            """
            A single training step
            """
            feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.input_y_domain: y_domain_batch, cnn.dropout_keep_prob: FLAGS.dropout_keep_prob}
            _, step, summaries, loss, accuracy, loss_d, accuracy_d = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.loss_domain, cnn.accuracy_domain],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}, loss-d {:g}, acc-d {:g}".format(time_str, step, loss, accuracy, loss_d, accuracy_d))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, y_domain_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.input_y_domain: y_domain_batch, cnn.dropout_keep_prob: 1.0}
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = read.batch_iter(
            list(zip(x_train, y_train, y_domain_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch, y_domain_batch = zip(*batch)
            train_step(x_batch, y_batch, y_domain_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, y_domain_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
