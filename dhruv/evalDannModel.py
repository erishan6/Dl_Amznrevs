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
# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "/Users/erishan6/Study WS17/CL-Deep Learning/Ishan_Repo/Dl_Amznrevs/dhruv/runs/music-music-1521116009/checkpoints/model-50", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
sequence_length = 200

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()

def createGloveEmbeddingsForTesting(domain):
    x_text, y = loadDataForDannGLOVE(domain, "train")
    # max_document_length = max([len(x.split(" ")) for x in x_text])
    # max_document_length = 200
    # vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    # x = np.array(list(vocab_processor.fit_transform(x_text)))
    # return x, y, len(vocab_processor.vocabulary_)
    wordsList = np.load('./SentimentClassification/wordsList.npy')
    print('Loaded the word list!')
    wordsList = wordsList.tolist() #  Originally loaded as numpy array
    wordsList = [word for word in wordsList] #  Encode words as UTF-8
    xs = []
    sentence_processed = 0
    for sentence in x_text:
        i=0
        firstSentence = np.zeros((sequence_length), dtype='int32')
        firstSentence[0] = wordsList.index("i")
        for word in sentence:
                if (i==len(sentence)-1 or i==200):
                    break;
                try:
                    firstSentence[i] = wordsList.index(word)
                except ValueError as e:
                    pass
                finally:
                    pass
                i=i+1
        xs.append(firstSentence)
        # if (sentence_processed%100==0):
            # print(sentence_processed)
        sentence_processed = sentence_processed+1
    x = np.array(xs)
    print(len(x))
    print(len(wordsList))
    # save as pickle

    f = open("./SentimentClassification/" + domain + ".pickle", 'wb')   # 'wb' instead 'w' for binary file
    pickle.dump([x, y, len(wordsList)], f, -1)       # -1 specifies highest binary protocol
    f.close()
    return x, y, len(wordsList)


def data(domains):
    xs = None
    ys = None
    size = 0
    base = "./SentimentClassification/"
    for domain in domains:
        filename = base + domain + ".pickle"
        if os.path.isfile(filename):
            f = open(filename, 'rb')   # 'rb' for reading binary file
            myarr = pickle.load(f)
            f.close()
            if xs:
                xs.append(myarr[0])
            else:
                xs = myarr[0]
            if ys:
                ys.append(myarr[1])
            else:
                ys = myarr[1]
            size = myarr[2]
        else:
            x, y, size = createGloveEmbeddingsForTesting(domain)
            if xs:
                xs.append(x)
            else:
                xs = x
            if ys:
                ys.append(y)
            else:
                ys = y
    return xs, ys, size

def evalDann():

    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    # CHANGE THIS: Load data. Load your own data here
    if FLAGS.eval_train:
        x_raw, y_test = read.loadDataForCNN(["dvd"],"test")
        y_test = np.argmax(y_test, axis=1)
    else:
        x_raw = ["a masterpiece four years in the making", "everything is off."]
        y_test = [1, 0]

    # Map data into vocabulary
    # vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    # vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    # x_test = np.array(list(vocab_processor.transform(x_raw)))
    x_test, y_test, v_test = data(["dvd"])
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
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = read.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
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
