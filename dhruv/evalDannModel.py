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

import logging
logging.basicConfig(filename='test.log',level=logging.DEBUG)

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


def compute_accuracy_count(pred, gold):
    count = 0
    for p, g in zip(pred, gold):
        if p[0] < p[1]:
            p = [0, 1]
        else:
            p = [1,0]
        if p == g:
            count += 1
    return count


def evalDann():

    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    x_test, y_test, v_test = data(["books"], "test", ".test.pickle")

    print("\nEvaluating...\n")
    # Evaluation
    # ==================================================
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    logging.info("############\nEvaluating for " + checkpoint_file)
    logging.info(FLAGS.checkpoint_dir)

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

            X = graph.get_operation_by_name("X").outputs[0]
            y = graph.get_operation_by_name("y").outputs[0]
            # domain = graph.get_operation_by_name("domain").outputs[0]
            train = graph.get_operation_by_name("train").outputs[0]
            predictions = graph.get_operation_by_name("label_predictor/prediction").outputs[0]

            # Generate batches for one epoch
            batches_x = read.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
            batches_y = read.batch_iter(list(y_test), FLAGS.batch_size, 1, shuffle=False)
            # Collect the predictions here
            all_predictions = [[0,0], [0,0]]
            for x_test_batch, y_test_batch in zip(batches_x,batches_y):
                # test_x = np.vstack([x_test_batch])
                # test_y = np.vstack([y_test_batch])
                # batch_predictions = sess.run(predictions,{X: test_x,y :test_y,train :False})
                batch_predictions = sess.run(predictions,{X: x_test_batch,y :y_test_batch,train:False})
                all_predictions = np.concatenate((all_predictions, batch_predictions),axis=0)

            all_predictions = all_predictions[2:]

    # Print accuracy if y_test is defined
    if y_test is not None:
        correct_predictions = compute_accuracy_count(all_predictions,y_test)
        logging.info("Total number of test examples: {}".format(len(y_test)))
        logging.info("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

    # Save the evaluation to a csv
    predictions_human_readable = np.column_stack((np.array(x_test), all_predictions))
    out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
    logging.info("Saving evaluation to {0}".format(out_path))
    with open(out_path, 'w') as f:
        csv.writer(f).writerows(predictions_human_readable)

evalDann()
logging.info("Ending evaluation\n############")
