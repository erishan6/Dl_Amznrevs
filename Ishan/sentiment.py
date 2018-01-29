import numpy as np
import tensorflow as tf
import re
from random import randint
import datetime


dirs = ['../SentimentClassification/en/books/'] #  , '../SentimentClassification/en/dvd/', '../SentimentClassification/en/music/']
files = ['pos.txt', 'neg.txt']
batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 1000
numDimensions = 50
maxSeqLength = 200
numReviews = 2000
wordsList = []
wordVectors = []


def saveword2vec(filepath):
	wordlist = []
	wordvectorlist = []
	with open(filepath,'r',encoding='utf-8') as f:
		line = f.readlines()
		for l in line:
			s = l.split()
			wordlist.append(s[0])
			wordvectorlist.append([float(i) for i in s[1:]])
	print (len(wordlist))
	wordvector = np.array(wordvectorlist)
	wordlistarray = np.array(wordlist)
	baseballIndex = wordlist.index('baseball')
	print (wordvector[baseballIndex])
	print (wordvector.shape)
	np.save("../SentimentClassification/glove/wordvector.npy", wordvector)
	np.save("../SentimentClassification/glove/wordlist.npy", wordlistarray)


def loadword2vec():
	wordsList = np.load('../SentimentClassification/glove/wordlist.npy')
	print('Loaded the word list!')
	print (wordsList)
	wordsList = wordsList.tolist() #  Originally loaded as numpy array
	wordsList = [word for word in wordsList] #  Encode words as UTF-8
	wordVectors = np.load('../SentimentClassification/glove/wordvector.npy')
	print ('Loaded the word vectors!')
	print (wordVectors)
	



def testEmbedding():
	
	loadword2vec()
	print(len(wordsList))
	print(wordVectors.shape)
	baseballIndex = wordsList.index('baseball')
	print (type(wordVectors[baseballIndex]))

	maxSeqLength = 10 #Maximum length of sentence
	numDimensions = 50 #Dimensions for each word vector
	firstSentence = np.zeros((maxSeqLength), dtype='int32')
	firstSentence[0] = wordsList.index("i")
	firstSentence[1] = wordsList.index("thought")
	firstSentence[2] = wordsList.index("the")
	firstSentence[3] = wordsList.index("movie")
	firstSentence[4] = wordsList.index("was")
	firstSentence[5] = wordsList.index("incredible")
	firstSentence[6] = wordsList.index("and")
	firstSentence[7] = wordsList.index("inspiring")
	#firstSentence[8] and firstSentence[9] are going to be 0
	print(firstSentence.shape)
	print(firstSentence) #Shows the row index for each word
	with tf.Session() as sess:
		print(tf.nn.embedding_lookup(wordVectors, firstSentence).eval().shape)


def cleanSentences(string):
	strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
	string = string.lower().replace("<br />", " ")
	return re.sub(strip_special_chars, "", string.lower())

def saveidmatrix():
	
	wordsList = np.load('../SentimentClassification/glove/wordlist.npy')
	print('Loaded the word list!')
	print (wordsList)
	wordsList = wordsList.tolist() #  Originally loaded as numpy array
	wordsList = [word for word in wordsList] #  Encode words as UTF-8
	wordVectors = np.load('../SentimentClassification/glove/wordvector.npy')
	print ('Loaded the word vectors!')
	print (wordVectors)
	
	fileCounter = 0
	ids = np.zeros((numReviews, maxSeqLength), dtype='int32')
	for d in dirs:
		for nf in files:
			with open(d + nf, "r", encoding='utf-8') as f:
				line = f.readlines()
				for l in line:
					indexCounter = 0
					cleanedLine = cleanSentences(l)
					split = cleanedLine.split()
					for word in split:
						try:
							ids[fileCounter][indexCounter] = wordsList.index(word)
						except ValueError:
							ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
						indexCounter = indexCounter + 1
						if indexCounter >= maxSeqLength:
							break
				fileCounter = fileCounter + 1
		np.save('idsMatrix.npy', ids)
	#Pass into embedding function and see if it evaluates. 

def loadidmatrix():
	ids = np.load('idsMatrix.npy')
	print (ids)



def getTrainBatch():
	labels = []
	arr = np.zeros([batchSize, maxSeqLength])
	for i in range(batchSize):
		if (i % 2 == 0):
			num = randint(1, 999)
			labels.append([1, 0])
		else:
			num = randint(1000, 1999)
			labels.append([0, 1])
		arr[i] = ids[num - 1:num]
	return arr, labels


def getTestBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(1000, 1999)
        if (num <= 959):
            labels.append([1, 0])
        else:
            labels.append([0, 1])
        arr[i] = ids[num - 1:num]
    return arr, labels


def tensorboardtest():
	tf.summary.scalar('Loss', loss)
	tf.summary.scalar('Accuracy', accuracy)
	merged = tf.summary.merge_all()
	logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
	writer = tf.summary.FileWriter(logdir, sess.graph)


#saveword2vec("../SentimentClassification/glove/glove_50d.txt")
#loadword2vec()
#saveidmatrix()
loadidmatrix()