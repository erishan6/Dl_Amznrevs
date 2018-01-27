import xml.etree.ElementTree as ET
import re

DATA_FOLDER = 'SentimentClassification/'
languages = ['en']#, 'fr', 'de']
domains = ['books', 'dvd', 'music']

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
                if float(item[1].text) == ( 1 or 2):
                    neg.write(re.sub( '\s+', ' ', (item[2].text).strip())+"\n")
                elif float(item[1].text) == ( 4 or 5):
                    pos.write(re.sub( '\s+', ' ', (item[2].text).strip())+"\n")
parsingData()
