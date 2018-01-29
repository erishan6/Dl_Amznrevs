import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import re
import numpy as np
# from gensim.models.doc2vec import LabeledSentence
# from gensim.models import Doc2Vec

DATA_FOLDER = 'SentimentClassification/'
languages = ['en']  # , 'fr', 'de']
# domains = ['books']  # , 'dvd', 'music']
# domains = ['dvd']  # , 'dvd', 'music']
domains = ['music']  # , 'dvd', 'music']

# filtetype can either be train or test. Rating is on index 1, review is on index 2.
def loadDataForCNN(filetype):
    xs = []
    ys = []
    for language in languages:
        for domain in domains:
            tree = ET.parse(DATA_FOLDER + language + '/' + domain + '/' + filetype + '.review')
            root = tree.getroot()
            for item in root:
                xs.append(preprocess(item[2].text))
                if(int(item[1].text[0]) < 3):
                    ys.append([0, 1])
                else:
                    ys.append([1, 0])
    return xs, ys


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


# filtetype can either be train or test. Rating is on index 1, review is on index 2.
def parsingData(filetype):
    for language in languages:
        for domain in domains:
            tree = ET.parse(DATA_FOLDER + language + '/' + domain + '/' + filetype + '.review')
            root = tree.getroot()
            return root


def doc2vec_source(train_size):
    f = open("d2v_train.txt", 'w')
    for language in languages:
        for domain in domains:
            tree = ET.parse(DATA_FOLDER + language + '/' + domain + '/' + 'train.review')
            root = tree.getroot()
            for item in root:
                f.write(item[2].text + "\n")
    f.close()


# def create_doc2vec_model(vectorsize):
#     sources = {'d2v_train.txt': 'TRAIN'}
#     sentences = LabeledSentence(sources)
#     cores = 4
#     model = Doc2Vec(min_count=1, window=10, size=vectorsize, sample=1e-4, negative=5, workers=cores, alpha=0.025, min_alpha=0.025)
#     model.build_vocab(sentences.to_array())
#     print('Starting to train...')
#     for epoch in range(10):
#         print('Epoch ', epoch)
#         model.train(sentences.sentences_perm())
#     model.save('./amazon.doc2vec')
#     return model


def visualizeData():
    numWords = []
    trainData = parsingData("train")
    numFiles = len(trainData)
    for item in trainData:
        counter = len(item[2].text.split())
        numWords.append(counter)
    print('The total number of files is', numFiles)
    print('The total number of words in the files is', sum(numWords))
    print('The average number of words in the files is', sum(numWords) / len(numWords))

    plt.hist(numWords, 50)
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.axis([0, 1200, 0, 8000])
    plt.show()


def preprocess(string):
    return re.sub(re.compile("[^A-Za-z0-9 ]+"), "", string.lower())


def computeidMatrix():
    wordsList = np.load('word2vec/wordsList.npy')
    print('Loaded the word list!')
    wordsList = wordsList.tolist()  # Originally loaded as numpy array
    wordsList = [word.decode('UTF-8') for word in wordsList]  # Encode words as UTF-8
    maxSeqLength = 250
    numFiles = 2000
    ids = np.zeros((numFiles, maxSeqLength), dtype='int32')
    fileCounter = 0
    trainData = parsingData("train")
    for item in trainData:
        indexCounter = 0
        cleanedLine = preprocess(item[2].text)
        split = cleanedLine.split()
        for word in split:
            try:
                ids[fileCounter][indexCounter] = wordsList.index(word)
            except ValueError:
                ids[fileCounter][indexCounter] = 399999  # Vector for unkown words
            indexCounter = indexCounter + 1
            if indexCounter >= maxSeqLength:
                break
        fileCounter = fileCounter + 1

    np.save('word2vec/idsMatrix', ids)


if __name__ == '__main__':
    computeidMatrix()
