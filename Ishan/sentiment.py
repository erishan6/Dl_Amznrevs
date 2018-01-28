import numpy as np
import tensorflow as tf

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
	wordsList = wordsList.tolist() #  Originally loaded as numpy array
	wordsList = [word for word in wordsList] #  Encode words as UTF-8
	wordVectors = np.load('../SentimentClassification/glove/wordvector.npy')
	print ('Loaded the word vectors!')
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




#saveword2vec("../SentimentClassification/glove/glove_50d.txt")
loadword2vec()