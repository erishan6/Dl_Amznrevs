import xml.etree.ElementTree as ET
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

DATA_FOLDER = 'SentimentClassification/'
languages = ['en']#, 'fr', 'de']
domains = ['books', 'dvd', 'music']

def parsingData():
	for language in languages:
		for domain in domains:
			tree = ET.parse(DATA_FOLDER + language + '/' + domain + '/' + 'train.review')
        	root = tree.getroot()
        	for item in root:
        		print(item[1].text)

def doc2vec_source(train_size):
    f = open("d2v_train.txt", 'w')
    for language in languages:
    	for domain in domains:
    		tree = ET.parse(DATA_FOLDER + language + '/' + domain + '/' + 'train.review')
        	root = tree.getroot()
        	for item in root:
        		f.write(item[2].text+"\n")
    f.close()

def create_doc2vec_model(vectorsize):
    sources = {'d2v_train.txt':'TRAIN'}
    sentences = LabeledLineSentence(sources)
    model = Doc2Vec(min_count=1, window=10, size=vectorsize, sample=1e-4, negative=5, workers=cores,alpha=0.025, min_alpha=0.025)
    model.build_vocab(sentences.to_array())
    print('Starting to train...')
    for epoch in range(10):
        print('Epoch ',epoch)
        model.train(sentences.sentences_perm())
    model.save('./amazon.doc2vec')
    return model

parsingData()
