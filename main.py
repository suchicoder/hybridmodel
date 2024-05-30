import numpy as np
import pandas as pd
import os, os.path, sys
import glob
from xml.etree import ElementTree
from bs4 import BeautifulSoup as bs
from bs4.element import Tag
import codecs
import nltk
from nltk import word_tokenize, pos_tag
from gensim.models import Word2Vec
from pso import ParticleSwarmOptimizedClustering
from utils import normalize

#this function appends all annotated files
def append_annotations(files):
    xml_files = glob.glob(files +"/*.xml")
    xml_element_tree = None
    new_data = ""
    for xml_file in xml_files:
        data = ElementTree.parse(xml_file).getroot()
        #print ElementTree.tostring(data)        
        temp = ElementTree.tostring(data)
        new_data+= str(temp)
    print("append_annotations")
    return(new_data)

#this function removes special characters and punctuations
def remov_punct(withpunct):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    without_punct = ""
    char = 'nan'
    for char in withpunct:
        if char not in punctuations:
            without_punct = without_punct + char
    return(without_punct)

files_path = "/home/ubuntu/crfnew/xml"

allxmlfiles = append_annotations(files_path)
soup = bs(allxmlfiles, "html5lib")

#identify the tagged element
docs = []
sents = []

for d in soup.find_all("document"):
   sents = []
   for wrd in d.contents:
    tags = []
    tags1= []
    NoneType = type(None)
    if isinstance(wrd.name, NoneType) == True:
        withoutpunct = remov_punct(wrd)
        temp = word_tokenize(withoutpunct)
        for token in temp:
            tags.append((token,'NA'))
    else:
        withoutpunct = remov_punct(wrd)
        temp = word_tokenize(withoutpunct)
        for token in temp:
            tags.append((token,wrd.name))
        sents = sents + tags
   docs.append(sents) #appends all the individual documents into one list 
print("append data")


data = []
for i, doc in enumerate(docs):
    tokens = [t for t, label in doc]
    tagged = nltk.pos_tag(tokens)
    data.append([(w, pos, label) for (w, label), (word, pos) in zip(doc, tagged)])


for i, doc in enumerate(docs):
    tokens = [t for t, label in doc]
    tagged = nltk.pos_tag(tokens)
    data.append([(w, pos, label) for (w, label), (word, pos) in zip(doc, tagged)])

#print(data)
token = []
token1 = []
v = []
v1 = []

# load model
model = Word2Vec.load('model.bin')
model1 = Word2Vec.load('model1.bin') 

def word2features(doc, i):
    word = doc[i][0]
    postag = doc[i][1]
    word_tag = doc[i][0]+'_'+doc[i][1]
    v.append(model[word])
    v1.append(model1[word_tag])


def extract_features(doc):
    print("extracting features")
    return [word2features(doc, i) for i in range(len(doc))]

X = [extract_features(doc) for doc in data]

for doc in data:
    for doc1 in doc:
        token = doc1[0]
        v.append(model[token])
        token1 = doc1[0] + "_" + doc1[1]
        v1.append(model1[token1])

a = np.asarray(v)
np.savetxt("vector.csv", a, delimiter="\t")

b = np.asarray(v1)
np.savetxt("tokenvector.csv", b, delimiter="\t")

print("vectors files created")

if __name__ == "__main__":
    data = pd.read_csv('vector.csv', sep='\t', header=None)
    # x = data.drop([1], axis=1)
    x = data.values
    x = normalize(x)
    pso = ParticleSwarmOptimizedClustering(
    n_cluster=6, n_particles=10, data=x, hybrid=True)  #, max_iter=2000, print_debug=50)
    cenlist = pso.run()
    a = np.asarray(cenlist)
    np.savetxt("centroid.csv", a, delimiter="\t")
    
    data = pd.read_csv('tokenvector.csv', sep='\t', header=None)
    # x = data.drop([1], axis=1)
    x = data.values
    x = normalize(x)
    pso = ParticleSwarmOptimizedClustering(
    n_cluster=6, n_particles=10, data=x, hybrid=True)  #, max_iter=2000, print_debug=50)
    cenlist = pso.run()
    a = np.asarray(cenlist)
    np.savetxt("tokencentroid.csv", a, delimiter="\t")

print("centoid files created")
