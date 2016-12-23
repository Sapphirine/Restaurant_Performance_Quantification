
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
from pyspark.sql.types import * 
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

#Module-level global variables for the `tokenize` function below
PUNCTUATION = set(string.punctuation)
STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

#Function to break text into "tokens", lowercase them, remove punctuation and stopwords, and stem them
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
    #stemmed = [STEMMER.stem(w) for w in no_stopwords]
    #return [w for w in stemmed if w]
    return [w for w in no_stopwords]

#Organize data 
data = d2_rdd \
    .map(lambda row: [row['name'],  \
                      row['text'] ])

#Take input from user  
print("Enter business name: ")
name=input("name: ")

print("Building topic model...")

#Filter data on the basis of business name
data = data \
    .filter(lambda row: row[0]==name) \
    .map(lambda row: row[1])

#Clean review text 
data_cleaned = data.map(lambda text: tokenize(text)).cache()

num_of_stop_words = 30      # Number of most common words to remove, trying to eliminate stop words
num_topics = 5              # Number of topics we are looking for
num_words_per_topic = 20    # Number of words to display for each topic
max_iterations = 100

termCounts = data_cleaned                             \
    .flatMap(lambda document: document)         \
    .map(lambda word: (word, 1))                \
    .reduceByKey( lambda x,y: x + y)            \
    .map(lambda tuple: (tuple[1], tuple[0]))    \
    .sortByKey(False)

#Get vocabulary
threshold_value = termCounts.take(num_of_stop_words)[num_of_stop_words - 1][0]    
vocabulary = termCounts                         \
    .filter(lambda x : x[0] < threshold_value)  \
    .map(lambda x: x[1])                        \
    .zipWithIndex()                             \
    .collectAsMap()

#Function which converts the given review into a vector of word counts    
def document_vector(document):
    id = document[1]
    counts = defaultdict(int)
    for token in document[0]:
        if token in vocabulary:
            token_id = vocabulary[token]
            counts[token_id] += 1
    counts = sorted(counts.items())
    keys = [x[0] for x in counts]
    values = [x[1] for x in counts]
    return (id, Vectors.sparse(len(vocabulary), keys, values))

documents = data_cleaned.zipWithIndex().map(document_vector).map(list)

#Form an inverted vocabulary
inv_voc = {value: key for (key, value) in vocabulary.items()}

#Converting the RDD into a dataframe
docdf = sql_context.createDataFrame(documents,['ind','features'])

#Building the model
lda = LDA(k=num_topics, maxIter=max_iterations)
lda_model = lda.fit(docdf)

res = lda_model.transform(docdf)

#Print results
topic_indices = lda_model.describeTopics(maxTermsPerTopic=num_words_per_topic)
for i in range(topic_indices.count()):
    print("\nTopic",str(i))
    for j in range(num_words_per_topic):
        print(inv_voc[topic_indices.select('termIndices').collect()[i][0][j]],":", topic_indices.select('termWeights').collect()[i][0][j])
print("\n\n",str(num_topics), "topics distributed over", str(documents.count()), "documents and",str(len(vocabulary)), "unique words\n")

df['id'] = range(0, len(df))
df=df.set_index(["id"])

#Calculate sentiments for sentences
selected_review=[['sentiment','text']]
for index in range(len(df)):
    raw_input=df.loc[index,'text']
    sent_tokenize_list = sent_tokenize(raw_input)
    for i in range(len(sent_tokenize_list)):
        t=tb(sent_tokenize_list[i])
        result=t.sentiment[0]
        if result>0.2 or result<-0.2:
            selected_review.append([result,sent_tokenize_list[i]])

#Save sentiments            
wr_file='file_sentiment.csv'
with open(wr_file,'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(selected_review) 

#Importing saved sentiments into dataframe
filename='file_sentiment.csv'
df = pd.read_csv(filename, encoding = "ISO-8859-1")
data2 =sql_context.createDataFrame(df)

#Converting dataframe to RDD
d2_rdd = data2.rdd

#Clean data 
data_cleaned = d2_rdd \
    .map(lambda line: (line['text'])) \
    .map(lambda text: tokenize(text)).cache()

data_avg_senti_cleaned = d2_rdd        \
    .map(lambda line: (tokenize(line['text']), line['sentiment']))     \
    .map(lambda tup: [(tup[0][i],(float(tup[1]),1)) for i in range(len(tup[0]))] )      

data_avg_senti_cleaned2=data_avg_senti_cleaned.flatMap(lambda line: line).cache()

#Calculate the positive and negative sentiments and frequency for each word 
data_avg_senti_cleaned = d2_rdd        \
    .map(lambda line: (tokenize(line['text']), line['sentiment']))     \
    .map(lambda tup: [(tup[0][i],(float(tup[1]),1)) for i in range(len(tup[0]))] )      \
    .flatMap(lambda line: line).cache()    \
    .map(lambda tup:(tup[0],(tup[1][0],1,0,0) if tup[1][0]>=0 else (0,0,tup[1][0],1) ))     \
    .reduceByKey( lambda x,y,: (x[0]+y[0],x[1]+y[1],x[2]+y[2],x[3]+y[3])  )        \
    .map(lambda tup: (tup[0], (tup[1][0]/tup[1][1] if tup[1][1]!=0 else 0 , tup[1][1],       \
    tup[1][2]/tup[1][3] if tup[1][3]!=0 else 0 ,tup[1][3] )))  \

#Build word2vec model     
print("Building word model.. ")

#Take input from user, find synonyms
while True:
    print("Enter word to find related terms: ")
    findTerm=input("word: ")

    model = Word2Vec().setMinCount(2).setVectorSize(100).fit(data_cleaned)

    res1=model.findSynonyms(findTerm, 5)
    res1_list=list(res1)        
        
    #find average positive and negative sentiment for top 5 synonyms 
    sent_vals={}# dictionary-> word: avg_pos,num_po,avg_neg,num_neg
    sent_vals_list=[]

    val=data_avg_senti_cleaned.lookup(findTerm)[0]
    sent_vals[findTerm]=val
    row=[findTerm,val[0], val[1],val[2],val[3]]
    sent_vals_list.append(row)

    for syn in res1_list:
        val=data_avg_senti_cleaned.lookup(syn[0])[0]
        sent_vals[syn[0]]=val
        row=[syn[0],val[0], val[1],val[2],val[3]]
        sent_vals_list.append(row)

    #print results    
    print("Word-Avg positive sentiment:Num positive sentiment, Avg negative sentiment:Num negative sentiment ")    
    for row in sent_vals_list:
        print(row[0], " - ", round(row[1],2), " : ", row[2], " , ", round(row[3],2), " : ", row[4])   
    
    print("Want to enter another word? [y/n]")
    opt=input("option: ")
    
    if opt=='n':
        break
