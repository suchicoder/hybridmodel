import glob
import re
from nltk.stem import WordNetLemmatizer
from nltk.parse.corenlp import CoreNLPDependencyParser

def append_txt(txtfiles):
        alltxt=[]
        read_files = glob.glob(txtfiles + "/*.txt")
        for f in read_files:
                with open(f, "rb") as infile:
                        alltxt.append(str(infile.read()))
        return alltxt

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
ef find_attribute(text, classes):
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

txtfile="/home/ubuntu/crfnew/text"
alltext = append_txt(txtfile)
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

label = find_labels('user')
 
