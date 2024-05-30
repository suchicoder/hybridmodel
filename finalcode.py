#invoke libraries

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from bs4 import BeautifulSoup as bs
from bs4.element import Tag
import pandas
import codecs
import statistics
import nltk
from nltk import word_tokenize, pos_tag
from sklearn.model_selection import train_test_split
import pycrfsuite
import os, os.path, sys
import glob
from xml.etree import ElementTree
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score
import glob
import re
from nltk.stem import WordNetLemmatizer
from nltk.parse.corenlp import CoreNLPDependencyParser
from sklearn.model_selection import RepeatedKFold

centroid_data = pd.read_csv('centroid.csv', sep='\t', header=None)
centroids = centroid_data.values
model = Word2Vec.load('model.bin')
print(model)

centroid_data = pd.read_csv('tokencentroid.csv', sep='\t', header=None)
tag_centroids = centroid_data.values
model1 = Word2Vec.load('model1.bin')
print(model1)
#vocabulary = model1.wv.vocab
#print(vocabulary)

def find_centroid(word):
    v = model[word]
    dist = 0
    for c in centroids:
        d = distance.euclidean(v, c)
        if dist == 0:
            dist  = 0
            centroid = c
        elif d <= dist:
            dist = d
            centroid = c
    print(centroid)
    return centroid

def find_tagcentroid(word_tag):
    v = model1[word_tag]
    dist = 0
    for c in tag_centroids:
        d = distance.euclidean(v, c)
        if dist == 0:
            dist = 0
            tag_centroid = c
        elif d <= dist:
            dist = d
            tag_centroid = c
    print(tag_centroid)
    return tag_centroid

def find_relation(text, classs, method, attribute, association, generalization):
    find=0
    subject = ''
    relation = []
    #print(text)
    dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')
    parse, = dep_parser.raw_parse(text)
    for governor, dep, dependent in parse.triples():
        #print(governor, dep, dependent)
        if dep == 'nsubj' or dep=='nsubjpass':
            find = 0
            det = str(dependent)
            subject = det.split('\'')[1]
            gov = str(governor)
            verb = gov.split('\'')[1]
            for x in classs: 
                if subject == x: 
                    for y in method:
                        if verb == y: 
                            relation.append(x + '-' + y)
                            break
                    for y in association:
                        if verb == y:
                            z = x + '-' + y
                            find = 1
                            break
                    for y in generalization:
                        if verb == y:
                            z = x + '-' + y
                            find = 1
                            break
 
                    break   
        if dep == 'dobj' and find ==1:
            det = str(dependent)
            objectt = det.split('\'')[1]
            relation.append(z + '-' + objectt)
            find = 0
            
        adet = str(dependent)
        asubject = adet.split('\'')[1]
        agov = str(governor)
        averb = agov.split('\'')[1]
            
        for n in attribute:           
            if n == asubject or n == averb:
                for x in classs:
                    if x == asubject or x == averb:
                        attribute.append(x + '-' + n)   
    return (relation, attribute)


def append_txt(txtfiles):
        alltxt=[]
        read_files = sorted(glob.glob(txtfiles + "/*.txt"))
        for f in read_files:
                with open(f, "rb") as infile:
                        alltxt.append(str(infile.read()))
        return alltxt, read_files

lemmatizer = WordNetLemmatizer()

def find_class(text):
    find=0
    find1=0
    class1=[]
    dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')
    parse, = dep_parser.raw_parse(text)
    for governor, dep, dependent in parse.triples():
        #print(governor, dep, dependent)
        if dep=='nsubj':
            det=str(dependent)
            subject=det.split('\'')
            if lemmatizer.lemmatize(subject[1].lower())!='system':
                class1.append(subject[1])
                find=1
        elif dep == 'dobj' and find!=1:
            det=str(dependent)
            objectt=det.split('\'')
            class1.append(objectt[1])

        if dep=='nsubjpass':
            gov=str(governor)
            verb=gov.split('\'')
            com=["identified", "recognized", "denoted"]
            for i in com:
                if verb[1].lower()==i:
                    subject=str(dependent).split('\'')
                    class1.append(subject[1])
        if dep=='cop':
            gov=str(governor)
            pattern = re.compile("nn*")
            if(pattern.match(gov.split('\'')[3])):
                class1.append(gov.split('\'')[1])

        if dep=='det'or dep=='amod':
            det=str(dependent)
            adj=det.split('\'')
            com=["any", "few", "many","each", "some", "several"]
            for i in com:
                if adj[1].lower()==i:
                    subject=str(governor).split('\'')
                    class1.append(subject[1])
        if dep=='nsubj'or dep=='nsubjpass':
            gov=str(governor)
            verb=gov.split('\'')
            com=["associated", "linked", "related"]
            for i in com:
                if verb[1].lower()==i:
                    subject=str(dependent).split('\'')
                    class1.append(subject[1])
                    find1=1
        if dep=='nmod' and find1==1:
            subject=str(dependent).split('\'')
            class1.append(subject[1])
        if dep=='mark' and find1==1:
            subject=str(governor).split('\'')
            class1.append(subject[1])
    return class1
    #print(class1)

def find_attribute(text, classes):
    attribute=[]
    finalattribute = []
    subj1=''
    dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')
    parse, = dep_parser.raw_parse(text)
    for governor, dep, dependent in parse.triples():
        #print(governor, dep, dependent)
        pattern = re.compile("nn*")
        if dep=='nmod:poss':
            att=str(governor).split('\'')
            attribute.append(att[1])
        if dep=='dobj':
            verb=str(governor).split('\'')
            comp=["enter","type","input"]
            for i in comp:
                if lemmatizer.lemmatize(verb[1].lower())==i and pattern.match(str(dependent).split('\'')[3]):
                    att=str(dependent).split('\'')
                    attribute.append(att[1])
        if dep=='case':
            adj=str(dependent).split('\'')
            comp=["on","for"]
            for i in comp:
                if adj[1].lower()==i and pattern.match(str(dependent).split('\'')[3]):
                    att=str(governor).split('\'')
                    attribute.append(att[1])
        if dep=='nmod':
            att=str(dependent).split('\'')
            attribute.append(att[1])

    for i in attribute:
        f=0
        for j in classes:
            if i.lower()==j.lower():
                f=1
        if f!=1:
            finalattribute.append(i)
    return finalattribute
    #print(attribute1)

def find_relation(text, classes):
    relation = []
    method = []
    rel=''
    flag1=0
    flag2=0
    class1=''
    class2=''
    dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')
    parse, = dep_parser.raw_parse(text)
    for governor, dep, dependent in parse.triples():
        #print(governor, dep, dependent)
        if dep=='nsubj':
            class1 = str(dependent).split('\'')[1]
            rel=str(governor).split('\'')[1]
        if dep=='nsubjpass':
            class1 = str(dependent).split('\'')[1]
        if dep=='dobj':
            if rel==str(governor).split('\'')[1]:
               class2 = str(dependent).split('\'')[1]
        for i in classes:
            if i.lower()==class1.lower():
               flag1=1
            if i.lower()== class2.lower():
               flag2=1
        if (flag1==1 and flag2==1):
            relation.append(rel)
        elif flag1==1 or flag2==0:
            method.append(rel)
    return (relation, method)

def find_rule(text):
    classes = find_class(text)
    att = find_attribute(text,classes)
    rel, meth = find_relation(text, classes)
    return (classes, att, meth, rel)

txtfile="/home/ubuntu/crfnew/extext"
alltext, read_files = append_txt(txtfile)
classes = []
attribute = []
method = []
relation = []

for text in alltext:
    classs, att, meth, rel = find_rule(text)
    classes.extend(classs)
    attribute.extend(att)
    method.extend(meth)
    relation.extend(rel)

#find the rules labels for each word
def find_labels(word):
    label=''
    for i in classes:
        if i == word:
            label = 'rclass'
            break
    for i in attribute:
        if i == word:
            label = 'rattribute'
            break
    for i in method:
        if i == word:
            label = 'rmethod'
            break
    for i in relation:
        if i == word:
            label = 'rrelation'
            break
    return label

def append_annotations(files):
    xml_files = sorted(glob.glob(files +"/*.xml"))
    xml_element_tree = None
    new_data = ""
    for xml_file in xml_files:
        data = ElementTree.parse(xml_file).getroot()
        #print ElementTree.tostring(data)        
        temp = ElementTree.tostring(data)
        new_data+= str(temp)
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

files_path = "/home/ubuntu/crfnew/exp"

allxmlfiles = append_annotations(files_path)
soup = bs(allxmlfiles, "html5lib")

#identify the tagged element
docs = []
sents = []

for d in soup.find_all("document"):
   sents = []  
   for wrd in d.contents:
    tags = []
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
labeled_data = []
for i, doc in enumerate(docs):
    tokens = [t for t, label in doc]
    tagged = nltk.pos_tag(tokens)
    data.append([(w, pos, label) for (w, label), (word, pos) in zip(doc, tagged)])

def word2features(doc, i):
    word = doc[i][0]
    postag = doc[i][1]
    word_tag = doc[i][0]+'_'+doc[i][1]
    label = find_labels(word)
    centroid = find_centroid(word)
    tag_centroid = find_tagcentroid(word_tag)
 
    # Common features for all words. You may add more features here based on your custom use case
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'label=' + label,
        'centroid='+ centroid,
        'tag_centroid=' + tag_centroid,
    ]

    # Features for words that are not at the beginning of a document
    if i > 0:
        word1 = doc[i-1][0]
        postag1 = doc[i-1][1]
        label1 = find_labels(word1)
        centroid1 = find_centroid(word)
        tag_centroid1 = find_tagcentroid(word_tag) 
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:word.isdigit=%s' % word1.isdigit(),
            '-1:postag=' + postag1,
            '-1:label=' + label1,
            '-1:centroid=' + centroid1,
            '-1:tag_centroid=' + tag_centroid1,

        ])
    else:
    # Indicate that it is the 'beginning of a document'
        features.append('BOS')
    # Features for words that are not at the end of a document
    if i < len(doc)-1:
        word1 = doc[i+1][0]
        postag1 = doc[i+1][1]
        label1 = find_labels(word1)
        centroid1 = find_centroid(word)
        tag_centroid1 = find_tagcentroid(word_tag)

        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:word.isdigit=%s' % word1.isdigit(),
            '+1:postag=' + postag1,
            '+1:label=' + label1,
            '+1:centroid=' + centroid1,
            '+1:tag_centroid=' + tag_centroid1,

        ])
    else:
        # Indicate that it is the 'end of a document'
        features.append('EOS')

    return features

# functions for extracting features in documents
def extract_features(doc):
    return [word2features(doc, i) for i in range(len(doc))]

def get_labels(doc):
    return [label for (token, postag, label) in doc]

def get_word(doc):
    return [token for (token, postag, label) in doc]

z = []
X = [extract_features(doc) for doc in data]
y = [get_labels(doc) for doc in data]
for text in alltext:
    z.append(text)
f = open('finalresults.txt','w')
f1 = open('relations.txt','w')
kf = RepeatedKFold(n_splits=2, n_repeats=1, random_state=None)

class_precision = []
method_precision = []
no_precision = []
attribute_precision = []
association_precision = []
generalization_precision = []

class_recall = []
attribute_recall = []
no_recall = []
method_recall = []
association_recall = []
generalization_recall = []

class_f1 = []
attribute_f1 = []
no_f1 = []
method_f1 = []
association_f1 = []
generalization_f1 = []

accuracy = []

for train, test in kf.split(X):
    X_train = []
    y_train = []
    z_train = []
    read_files_train = []
    X_test = []
    y_test = []
    z_test = []
    read_files_test = []
    for i in train:
        X_train.append( X[i])
        y_train.append( y[i])
        z_train.append( z[i])
        read_files_train.append( read_files[i])   
    for i in test:
        X_test.append( X[i])
        y_test.append( y[i])
        z_test.append( z[i])
        read_files_test.append( read_files[i])

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    trainer = pycrfsuite.Trainer(verbose=True)
    # Submit training data to the trainer
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)
    
    # Set the parameters of the model
    trainer.set_params({
        # coefficient for L1 penalty
        'c1': 0.1,

        # coefficient for L2 penalty
        'c2': 0.01,  

        # maximum number of iterations
        'max_iterations': 200,

        # whether to include transitions that
        # are possible, but not observed
        'feature.possible_transitions': True
    })

    # Provide a file name as a parameter to the train function, such that
    # the model will be saved to the file when training is finished
    trainer.train('crf.model')

    tagger = pycrfsuite.Tagger()
    tagger.open('crf.model')
    y_pred = [tagger.tag(xseq) for xseq in X_test]
     
    # Let's take a look at a random sample in the testing set
    i =0
    classs1 = []
    method1 = []
    attribute1 = []
    no1 = []
    generalization1 = []
    association1 = [] 
    
    for xi, yi, zi, readi  in zip(X_test, y_test, z_test, read_files_test): 
        classs1 = []
        method1 = []
        attribute1 = []
        no1 = []
        generalization1 = []
        association1 = []

        for y1, x1 in zip(yi, [x[1].split("=")[1] for x in xi]): 
            #for y1, x1 in zip(y_pred[i], [x[1].split("=")[1] for x in X_test[i]]):
            #print("%s (%s)" % (y1, x1))
            
            if y1 == 'class':
                classs1.append(x1)
            if y1 == 'method':
                method1.append(x1) 
            if y1 == 'attribute':
                attribute1.append(x1)
            if y1 == 'no':
                no1.append(x1)
            if y1 == 'generalization':
                generalization1.append(x1)
            if y1 == 'association':
                association1.append(x1)     
        relationlist, attributelist = find_relation(zi, classs1, method1, attribute1, association1, generalization1)
        f1.write(readi)
        f1.write('\n')
        f1.write("relation*******************")
        f1.write('\n')
        for rel in relationlist:
            f1.write(rel)
            f1.write('\n')
        f1.write("attribute*****************") 
        f1.write('\n')
        for att in attributelist:
            f1.write(att)  
            f1.write('\n')
    # Create a mapping of labels to indices
    labels = {"class": 0, "attribute": 1, "association": 2, "method": 3, "no": 4, "generalization": 5}

    # Convert the sequences of tags into a 1-dimensional array
    predictions = np.array([labels[tag] for row in y_pred for tag in row])
    truths = np.array([labels[tag] for row in y_test for tag in row])
    
    # Print out the classification report
    f.write(classification_report(truths, predictions, target_names=["class", "attribute", "association", "method", "no", "generalization"]))
    
    class_precision.append(precision_score(truths,predictions,average='macro'))
    attribute_precision.append(precision_score(truths,predictions,average='macro'))
    no_precision.append(precision_score(truths,predictions,average='macro'))
    method_precision.append(precision_score(truths,predictions,average='macro'))
    association_precision.append(precision_score(truths,predictions,average='macro'))
    generalization_precision.append(precision_score(truths,predictions,average='macro'))  
     
    class_recall.append(recall_score(truths,predictions,average='macro'))
    attribute_recall.append(recall_score(truths,predictions,average='macro')) 
    no_recall.append(recall_score(truths,predictions,average='macro'))
    method_recall.append(recall_score(truths,predictions,average='macro'))
    association_recall.append(recall_score(truths,predictions,average='macro'))
    generalization_recall.append(recall_score(truths,predictions,average='macro'))
   
    class_f1.append(f1_score(truths,predictions,average='macro'))
    attribute_f1.append(f1_score(truths,predictions,average='macro'))
    no_f1.append(f1_score(truths,predictions,average='macro'))
    method_f1.append(f1_score(truths,predictions,average='macro'))
    association_f1.append(f1_score(truths,predictions,average='macro'))
    generalization_f1.append(f1_score(truths,predictions,average='macro')) 

    accuracy.append(accuracy_score(truths,predictions))

f.write('\n avg_class_precision = '+ str( statistics.mean(class_precision)))
f.write('\n avg_attribute_precision  = '+ str( statistics.mean(attribute_precision)))
f.write('\n avg_no_precision = '+ str( statistics.mean(no_precision)))
f.write('\n avg_method_precision = '+ str( statistics.mean(method_precision)))
f.write('\n avg_association_precision = '+ str( statistics.mean(association_precision)))
f.write('\n avg_generalization_precision = '+ str( statistics.mean(generalization_precision)))
f.write('\n')
f.write('\n avg_class_recall = '+ str( statistics.mean(class_recall)))
f.write('\n avg_attribute_recall  = '+ str( statistics.mean(attribute_recall)))
f.write('\n avg_no_recall = '+ str( statistics.mean(no_recall)))
f.write('\n avg_method_recall = '+ str( statistics.mean(method_recall)))
f.write('\n avg_association_recall = '+ str( statistics.mean(association_recall)))
f.write('\n avg_generalization_recall = '+ str( statistics.mean(generalization_recall)))
f.write('\n')
f.write('\n avg_class_f1 = '+ str( statistics.mean(class_f1)))
f.write('\n avg_attribute_f1  = '+ str( statistics.mean(attribute_f1)))
f.write('\n avg_no_f1 = '+ str( statistics.mean(no_f1)))
f.write('\n avg_method_f1 = '+ str( statistics.mean(method_f1)))
f.write('\n avg_association_f1 = '+ str( statistics.mean(association_f1)))
f.write('\n avg_generalization_f1 = '+ str( statistics.mean(generalization_f1)))
f.write('\n')
f.write('\n avg_accuracy = '+ str( statistics.mean(accuracy)))

f.close()
f1.close()
