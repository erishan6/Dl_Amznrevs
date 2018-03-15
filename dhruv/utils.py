from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from read import loadDataForDannGLOVE
import os
import pickle

# config settings,make sure that they are same as in DannModel
# TODO: Need to move this into a common file later
sequence_length = 200

# static strings
BASE_DATA_DIR = "./SentimentClassification/"

# Model construction utilities below adapted from
# https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html#deep-mnist-for-experts
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]


def batch_generator(data, batch_size, shuffle=False):
    """Generate batches of data.

    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    # print(data[0])
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]


def imshow_grid(images, shape=[2, 8]):
    """Plot images in a grid of a given shape."""
    fig = plt.figure(1)
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)

    size = shape[0] * shape[1]
    for i in range(size):
        grid[i].axis('off')
        grid[i].imshow(images[i])  # The AxesGrid object work as a list of axes.

    plt.show()


def plot_embedding(X, y, d, title=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.bwr(d[i] / 1.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


def createGloveEmbeddings(domain, dataset, file_suffix):
    x_text, y = loadDataForDannGLOVE(domain, dataset)
    # max_document_length = max([len(x.split(" ")) for x in x_text])
    # max_document_length = 200
    # vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    # x = np.array(list(vocab_processor.fit_transform(x_text)))
    # return x, y, len(vocab_processor.vocabulary_)
    wordsList = np.load(BASE_DATA_DIR + 'wordsList.npy')
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

    f = open(BASE_DATA_DIR + domain + file_suffix, 'wb')   # 'wb' instead 'w' for binary file
    pickle.dump([x, y, len(wordsList)], f, -1)       # -1 specifies highest binary protocol
    f.close()
    return x, y, len(wordsList)


def data(domains, dataset, file_suffix):
    xs = None
    ys = None
    size = 0
    base = BASE_DATA_DIR
    for domain in domains:
        filename = base + domain + file_suffix
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
            x, y, size = createGloveEmbeddings(domain, dataset, file_suffix)
            if xs:
                xs.append(x)
            else:
                xs = x
            if ys:
                ys.append(y)
            else:
                ys = y
    return xs, ys, size


if __name__ == '__main__':
    data(["music", "books", "dvd"], "test", ".test.pickle")
