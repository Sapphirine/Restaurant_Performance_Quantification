
#Instructions
#1. Give SPARK_HOME path (line 11) 
#2. Enter the path for the csv dataset file in filename (line 50)  

import findspark
import os

#set environment variables 
os.environ["PYTHONHASHSEED"]="0"  
os.environ["SPARK_HOME"] ="path-to-spark"
#check 
x=os.environ.get('SPARK_HOME', None)
print(x)

findspark.init()

import pyspark

from collections import defaultdict
from pyspark import SparkContext
from pyspark.ml.linalg import Vector, Vectors
from pyspark.ml.clustering import LDA, LDAModel
from pyspark.sql import SQLContext
import re
from pyspark.mllib.feature import Word2Vec

import string 

import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from pyspark.mllib.feature import HashingTF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes

from textblob import TextBlob as tb
import pandas as pd
import csv

from pprint import pprint

sc = SparkContext('local', 'TestJSON')
sc.setLogLevel("ERROR")
sql_context = SQLContext(sc)

filename='path-to-csv-file'
df = pd.read_csv(filename, encoding = "ISO-8859-1")

#in case categories turn into string of lists then use the following code to convert it back to lists 
def cut(y):
    return [x.strip() for x in eval(y)]        

df.categories = df.categories.apply(cut)

data2 =sql_context.createDataFrame(df)

#Converting dataframe to RDD
d2_rdd = data2.rdd

# Module-level global variables for the `tokenize` function below
PUNCTUATION = set(string.punctuation)
STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

# Function to break text into "tokens", lowercase them, remove punctuation and stopwords, and stem them
def tokenize(text):
    tokens = word_tokenize(text)
    lowercased = [t.lower() for t in tokens]
    alphaed = [x for x in lowercased if x.isalpha()]
    shorted = [x for x in alphaed if len(x) > 3]
    no_punctuation = []
    for word in shorted:
        punct_removed = ''.join([letter for letter in word if not letter in PUNCTUATION])
        no_punctuation.append(punct_removed)
    no_stopwords = [w for w in no_punctuation if not w in STOPWORDS]
    return [w for w in no_stopwords]

#step1: organize data 
data = d2_rdd \
    .map(lambda row: [row['business_id'],  \
                      sent_tokenize(row['text']), \
                      (row['categories'], row['Metro Area'], row['name'], row['latitude'], row['longitude'] )])


#take input from user  
print("Enter query filter: Enter 'None' if N/A")
metro_area=input("metro_area: ")
category=input("category: ")

data_filtered=data

if metro_area!='None':
    #step 2: filter metro area based on query
    data_filtered=data \
        .filter(lambda row: (row[2][1]==metro_area))

if category!='None':
    #step 2: filter category based on query
    data_filtered=data \
        .filter(lambda row: (category in row[2][0]))

#step 3: create a map of details needed 
details=data_filtered \
    .map(lambda row: (row[0], (row[2][0],row[2][1],row[2][2],row[2][3],row[2][4] ) )) \
    .reduceByKey(lambda x,y: (y))

#step 4: create a vocabulary 
vocab = data_filtered \
    .map(lambda row: (row[1])) \
    .flatMap(lambda text: text) \
    .map(lambda text: tokenize(text))

print("Building model...")
#step 5: create word2vec model
model = Word2Vec().setMinCount(2).setVectorSize(100).fit(vocab)

print("Enter query terms (0 for no more query) :")
findTerm=[]

while True:
    x=input("term: ")
    if x!='0':
        findTerm.append(x)
    else:
        break

related_terms=[]

for i in range(len(findTerm)):
    print("searching for matches for ",findTerm[i])
    related_terms=related_terms+(list(model.findSynonyms(findTerm[i], 5)))
    related_terms.append((findTerm[i],1))

related_terms_words=[tup[0] for tup in related_terms]
print("Found related words... ")
pprint(related_terms_words)

#step 7: create a sentence map  
data_sentence_map=data_filtered \
    .map(lambda row: [( row[0], word_tokenize(row[1][i].lower())) for i in range(len(row[1]))] ) \
    .flatMap(lambda tup: tup)

#step 8: filter on the basis of word occurence 
data_word_filtered=data_sentence_map \
    .filter(lambda tup: any(word in tup[1] for word in related_terms_words ))

#step 9: find sentiment, remove sentence (not needed now)
data_sentiments=data_word_filtered \
    .map(lambda tup: (tup[0], tb(' '.join(tup[1])).sentiment[0] ))

#step 10: filter sentiments
data_sentiments_filtered=data_sentiments \
    .filter(lambda tup: tup[1]>0.2 or tup[1]< -0.2)

#step 11: seperate positive and negative sentiments 
data_sentiments_filtered=data_sentiments_filtered \
    .map(lambda tup: (tup[0], \
                      (tup[1],1,0,0) if tup[1]>=0 \
                       else (0,0,tup[1],1) ))

#step 12: add sentiments
data_sentiments_grouped=data_sentiments_filtered \
    .reduceByKey(lambda x,y: (x[0]+y[0],x[1]+y[1],x[2]+y[2],x[3]+y[3] ))

#step 13: average sentiments
data_sentiments_grouped=data_sentiments_grouped \
    .map(lambda tup: (tup[0], (tup[1][0]/tup[1][1] if tup[1][1]!=0 else 0 , tup[1][1], \
    tup[1][2]/tup[1][3] if tup[1][3]!=0 else 0 ,tup[1][3] )))

#step 14: prepare results
data_sentiments_grouped=data_sentiments_grouped \
    .map(lambda tup: [tup[0],tup[1][0],tup[1][1],tup[1][2],tup[1][3]])

print("Enter minimum positive reviews to consider: " )
threshold=input("threshold: ")
print("Preparing results...")
data_sentiments_grouped=data_sentiments_grouped \
    .filter(lambda row: row[2]>=int(threshold))
    
result=data_sentiments_grouped.collect()

final_result=[]
for i in range(len(result)):
    temp=details.lookup(result[i][0])[0]
    final_result.append([result[i][0],result[i][1],result[i][2],result[i][3],result[i][4], \
                         temp[0],temp[1],temp[2],temp[3],temp[4]])

final_result.sort(key=lambda x: x[2])

print("Loading query results:")
pprint("Name, Average positive sentiment, Number of positive sentiments")
for row in final_result:
    print(row[7], " , ", round(row[1],2), " , ", row[2])



