rom gensim.models import KeyedVectors
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import os, os.path, sys
import glob
import numpy as np
import pandas as pd
import nltk
from pso import ParticleSwarmOptimizedClustering
from utils import normalize

def append_txt(txtfiles):
        alltxt=[]
        read_files = glob.glob(txtfiles + "/*.txt")
        for f in read_files:
                with open(f, "rb") as infile:
                        alltxt.append(str(infile.read()))
                       # print(str(infile.read())) 
        return alltxt

txtfile="/home/ubuntu/crfnew/text" #path for text files
alltext = append_txt(txtfile)

def remov_punct(withpunct):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    without_punct = ""
    char = 'nan'
    for char in withpunct:
        if char not in punctuations:
            without_punct = without_punct + char
            #print without_punct
    return(without_punct)

# load model
model = Word2Vec.load('model.bin')
model1 = Word2Vec.load('model1.bin')

v = []
token1 = []
for text in alltext:
    for i in word_tokenize(text):
        j = remov_punct(i)
        v.append(model[j])
    tagged=[]
    for i in word_tokenize(text):
        k = remov_punct(i)
        if k:
            tagged.append(nltk.pos_tag([k]))

    for i in tagged:
        st = str(i)
        sub = st[2:-2].replace("'","").replace(",","_")
        token1.append(model1[sub])

a = np.asarray(v)
np.savetxt("vector.csv", a, delimiter="\t")

b = np.asarray(token1)
np.savetxt("tokenvector.csv", b, delimiter="\t")

