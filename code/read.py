import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import re
import numpy as np
import tensorflow as tf
import pickle
# from gensim.models.doc2vec import LabeledSentence
# from gensim.models import Doc2Vec

DATA_FOLDER = 'SentimentClassification/'
languages = ['en']  # , 'fr', 'de']
domains = ['books']#  , 'dvd', 'music']
# domains = ['dvd']  # , 'dvd', 'music']
#domains = ['music']  # , 'dvd', 'music']

# filtetype can either be train or test. Rating is on index 1, review is on index 2.
def loadDataForCNN(domains, filetype):
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


def loadDataForDannGLOVE(domain, filetype):
    xs = []
    ys = []
    for language in languages:
        tree = ET.parse(DATA_FOLDER + language + '/' + domain + '/' + filetype + '.review')
        root = tree.getroot()
        for item in root:
            xs.append(preprocess(item[2].text))
            if(int(item[1].text[0]) < 3):
                ys.append([0, 1])
            else:
                ys.append([1, 0])
    return xs, ys

def loadDataForDANN(domains, filetype):
    xs = []
    ys = []
    for language in languages:
        for domain in domains:
            tree = ET.parse(DATA_FOLDER + language + '/' + domain + '/' + filetype + '.review')
            root = tree.getroot()
            for item in root:
                xs.append(preprocess(item[2].text))

                if(int(item[1].text[0]) < 3):
                    ys.append(0)
                else:
                    ys.append(1)
    return xs, ys

def saveword2vec(filepath):
    wordsList = []
    wordsVectors = []
    with open(filepath,'r',encoding='utf-8') as f:
        line = f.readlines()
        for l in line:
            s = l.split()
            wordsList.append(s[0])
            wordsVectors.append([float(i) for i in s[1:]])
    print (len(wordsList))
    wordsVectors = np.array(wordsVectors)
    wordsListArray = np.array(wordsList)
    baseballIndex = wordsList.index('baseball')
    print (wordsVectors[baseballIndex])
    print (wordsVectors.shape)
    np.save("../SentimentClassification/glove/wordsVectors.npy", wordsVectors)
    np.save("../SentimentClassification/glove/wordsList.npy", wordsListArray)


def loadword2vec():
    wordsList = np.load('../SentimentClassification/glove/wordsList.npy')
    print('Loaded the word list!')
    print (wordsList)
    wordsList = wordsList.tolist() #  Originally loaded as numpy array
    wordsList = [word for word in wordsList] #  Encode words as UTF-8
    wordsVectors = np.load('../SentimentClassification/glove/wordsVectors.npy')
    print ('Loaded the word vectors!')
    print (wordsVectors)

def testEmbedding():
    wordsList = np.load('../SentimentClassification/glove/wordsList.npy')
    print('Loaded the word list!')
    print (wordsList)
    wordsList = wordsList.tolist() #  Originally loaded as numpy array
    wordsList = [word for word in wordsList] #  Encode words as UTF-8
    wordsVectors = np.load('../SentimentClassification/glove/wordsVectors.npy')
    print ('Loaded the word vectors!')
    print (wordsVectors)
    print(len(wordsList))
    print(wordsVectors.shape)
    baseballIndex = wordsList.index('baseball')
    print (type(wordsVectors[baseballIndex]))

    maxSeqLength = 10 #Maximum length of sentence
    numDimensions = 300 #Dimensions for each word vector
    firstSentence = np.zeros((maxSeqLength), dtype='int32')
    firstSentence[0] = wordsList.index("i")
    #firstSentence[8] and firstSentence[9] are going to be 0
    print(firstSentence.shape)
    print(firstSentence) #Shows the row index for each word
    with tf.Session() as sess:
        print(tf.nn.embedding_lookup(wordsVectors, firstSentence).eval().shape)



def create_embedding_vectors(filetype):
    maxSeqLength = 200 #Maximum length of sentence
    numDimensions = 300 #Dimensions for each word vector
    wordsList = np.load('../SentimentClassification/glove/wordsList.npy')
    print('Loaded the word list!')
    # print (wordsList)
    wordsList = wordsList.tolist() #  Originally loaded as numpy array
    wordsList = [word for word in wordsList] #  Encode words as UTF-8
    wordsVectors = np.load('../SentimentClassification/glove/wordsVectors.npy')
    print ('Loaded the word vectors!')
    for language in languages:
        for domain in domains:
            xs = []
            ys = []
            tree = ET.parse(DATA_FOLDER + language + '/' + domain + '/' + filetype + '.review')
            root = tree.getroot()
            output_embedding = open(DATA_FOLDER + language + '/' + domain + '/data.pkl', 'wb')
            y=0
            for item in root:
                review = preprocess(item[2].text).split()
                firstSentence = np.zeros((maxSeqLength), dtype='int32')
                i = 0
                print (y)
                y=y+1
                for word in review:
                    if (i==len(review)-1 or i==200):
                        break;
                    try:
                        firstSentence[i] = wordsList.index(word)
                    except ValueError as e:
                        pass
                    finally:
                        pass
                    i=i+1
                    # print(firstSentence) #Shows the row index for each word
                with tf.Session() as sess:
                    # print(tf.nn.embedding_lookup(wordsVectors, firstSentence).eval().shape)
                    xs.append(tf.nn.embedding_lookup(wordsVectors, firstSentence).eval().shape)
                if(int(item[1].text[0]) < 3):
                    ys.append(0)
                else:
                    ys.append(1)
            data1=[xs,ys]
            print (data1)
            # pickle.dump(data1, output_embedding)
            output_embedding.close()


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
    #computeidMatrix()
    create_embedding_vectors("train")
