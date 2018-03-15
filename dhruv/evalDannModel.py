#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import read
from read import loadDataForDannGLOVE
from tensorflow.contrib import learn
import csv
import pickle
from utils import data
# Parameters
# ==================================================

# Data Parameters

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
sequence_length = 200

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()


def evalDann():

    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    x_test, y_test, v_test = data(["dvd"], "test", ".test.pickle")

    print("\nEvaluating...\n")
    # Evaluation
    # ==================================================
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    print(checkpoint_file)
    print(FLAGS.checkpoint_dir)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            # Compute final evaluation on test data
            # self.X = tf.placeholder(tf.int32, [None, sequence_length], name="X")
            # self.y = tf.placeholder(tf.float32, [None, 2], name="y")
            # self.domain = tf.placeholder(tf.float32, [None, 2], name="domain")
            # self.l = tf.placeholder(tf.float32, [], name="l")
            # self.train = tf.placeholder(tf.bool, [], name="train")
            # source_acc = sess.run(label_acc, feed_dict={model.X: X, model.y: y, model.domain: domain_labels, model.train: False})

            X = graph.get_operation_by_name("X").outputs[0]
            y = graph.get_operation_by_name("y").outputs[0]
            print(X)
            print(y)
            # input_y = graph.get_operation_by_name("input_y").outputs[0]

            # Tensors we want to evaluate
            # predictions = graph.get_operation_by_name("label_acc").outputs[0]

            # Generate batches for one epoch
            batches = read.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = sess.run({X: x_test_batch,y:1})
                print(batch_predictions)
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    # Print accuracy if y_test is defined
    if y_test is not None:
        correct_predictions = float(sum(all_predictions == y_test))
        print("Total number of test examples: {}".format(len(y_test)))
        print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

    # Save the evaluation to a csv
    predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
    out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
    print("Saving evaluation to {0}".format(out_path))
    with open(out_path, 'w') as f:
        csv.writer(f).writerows(predictions_human_readable)

evalDann()
