import numpy as np


def loadword2vec(filepath):
	wordlist = []
	wordvector = []
	with open(filepath,'r',encoding='utf-8') as f:
		line = f.readlines()
		for l in line:
			s = l.split()
			wordlist.append(s[0])
			wordvector.append(s[1:])
	print (len(wordlist))
	x = np.array(wordvector) #. .dump(open('wordvector.npy', 'wb'))
	y = np.array(wordlist) #. .dump(open('wordList.npy', 'wb'))
	# print (type(x))
	# print (type(y))
	baseballIndex = wordlist.index('baseball')
	print (x[baseballIndex])
	print (x.shape)


loadword2vec("../SentimentClassification/glove/glove_50d.txt")