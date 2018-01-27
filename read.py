import xml.etree.ElementTree as ET
import re
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
#%matplotlib inline

DATA_FOLDER = 'SentimentClassification/'
languages = ['en']#, 'fr', 'de']
domains = ['books', 'dvd', 'music']
dirs = ['SentimentClassification/en/books/','SentimentClassification/en/dvd/', 'SentimentClassification/en/music/']
files = ['pos.txt','neg.txt']

def parsingData():
    for language in languages:
        for domain in domains:
            tree_head  =  DATA_FOLDER + language + '/' + domain + '/'
            pos = open(tree_head+"pos.txt", 'w')
            neg = open(tree_head+"neg.txt", 'w')
            tree = ET.parse(tree_head+'train.review')
            root = tree.getroot()
            for item in root:
                #print(item[1].text)
                if int(float(item[1].text)) == 1 or int(float(item[1].text)) == 2 :
                    neg.write(re.sub( '\s+', ' ', (item[2].text).strip())+"\n")
                elif int(float(item[1].text)) == 4 or int(float(item[1].text)) == 5 :
                    pos.write(re.sub( '\s+', ' ', (item[2].text).strip())+"\n")

def plotGraph(numWords):
    plt.hist(numWords, 50)
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.axis([0, 1400, 0, 2100])
    plt.show()

def main():
    # positiveFiles = ['positiveReviews/' + f for f in listdir('positiveReviews/') if isfile(join('positiveReviews/', f))]
    # negativeFiles = ['negativeReviews/' + f for f in listdir('negativeReviews/') if isfile(join('negativeReviews/', f))]
    numWords = []
    numSentences = 0
    for d in dirs:
        for nf in files:
            with open(d+nf, "r", encoding='utf-8') as f:
                line=f.readlines()
                for l in line:
                    counter = len(l.split(" "))
                    numWords.append(counter)

    print('The total number of sentences is', len(numWords))
    print('The total number of words in the all sentences is', sum(numWords))
    print ('len numWords = ', len(numWords))
    print('The average number of words in the sentences is', sum(numWords)/len(numWords))
    plotGraph(numWords)

if __name__=="__main__":
    main()
    # parsingData()