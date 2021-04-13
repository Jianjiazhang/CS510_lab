# import json
import numpy as np
import jsonlines
#import emoji
import nltk
import string
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
# for stem words

# genrate stop-word dictionary inorder to remove useless words and charaters
stop_words = stopwords.words('english')
# @USER can't remove
stop_words.append('@USER')
stop_words = set(stop_words)
exclude = set(string.punctuation)
stop_words = stop_words.union(stop_words, exclude)
stop_words = list(stop_words)

dataStep1 = []
ps = PorterStemmer()
with jsonlines.open('data/train.jsonl') as reader:
    for item in reader:
        label = item["label"]
        # store string data into temp object
        temp = item["response"].split()
        # emoji.demojize(temp)
        temp = [ps.stem(word) for word in temp if word.lower() not in stop_words]
        response = ' '.join(temp).lower()
        # store string data into temp object
        temp = item["context"][0].split()
        #emoji.demojize(temp)
        temp = [ps.stem(word)for word in temp if word.lower() not in stop_words]
        context = ' '.join(temp).lower()
        dataStep1.append([label,response,context])
# build the vocabulary
vocabulary = set()
for eachLineAsList in dataStep1:
    vocabulary = vocabulary.union(eachLineAsList[1].split())
vocabulary = list(vocabulary)
#vocabulary size
d = len(vocabulary)
# print(d)
n = 5000

term_doc_matrix = np.zeros((n, d))
for i in range(n):
    for j in range(d):
        term_doc_matrix[i][j] = dataStep1[i][1].count(vocabulary[j])

print(term_doc_matrix.shape)

# count each words for each line
# csv
# np.savetxt("term_doc_matrix.csv", term_doc_matrix)
