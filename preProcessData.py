# import json
import numpy as np
import jsonlines
import nltk
import string
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
# for preProcessingString
import re
from autocorrect import spell
# for translate
import goslate
from deep_translator import GoogleTranslator
# for write the file
import csv


# genrate stop-word dictionary inorder to remove useless words and charaters
stop_words = stopwords.words('english')
# stop_words = set(stop_words)
# exclude = set(string.punctuation)
# stop_words = stop_words.union(stop_words, exclude)
# stop_words = list(stop_words)

RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
dataStep1 = []
dataStep2 = []
ps = PorterStemmer()

# pre precess string
def preProcessingString(text):
    # remove emoji
    # text = RE_EMOJI.sub(r'', text)
    # lower
    text = ' '.join([word.lower() for word in word_tokenize(text)])

    # to delete the word like "PAN-00049042"
    # text = text.split("-")
    # text = ' '.join(str for str in text)

    # remove numbers, punctuation, strange character
    text = re.sub("[^a-zA-Z ]+", "", text)
    # stem and remove words in stopwords
    
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    text = ' '.join(text)
    lemmatizer = WordNetLemmatizer()
    text = lemmatizer.lemmatize(text)
    # spell check: correct the words like 'wrld' to 'world' (note: take longer time)
    # text = [spell(word) for word in (nltk.word_tokenize(text))]
    # text = ' '.join(text)
    return text



def getStringBasedOnStringOrList(text):
    if (text is None):
      return " "
    gs = goslate.Goslate()
    if (type(text) == list):
        text = ' '.join(str for str in text)
    # translate when needed
    # text = GoogleTranslator(source='auto', target='en').translate(text)
    # language_id = gs.detect(text)
    # if (language_id == 'German'):
    #     text = gs.translate(text, "en")
    text = preProcessingString(text)
    return text

i = 0
# get a n * 4 matrix from dataset
with jsonlines.open('../datasets/dataset.jsonl') as reader:
    for item in reader:
        i = i + 1
        # if (i > 100):
        #     break
        print("current: ", i)
        doc_id = item["id"]
        print(doc_id)
        print(item["title"])
        title = getStringBasedOnStringOrList(item["title"])
        abstract = getStringBasedOnStringOrList(item["abstract"])
        topic = getStringBasedOnStringOrList(item["topic"])
        dataStep1.append([doc_id, title, abstract, topic])
        print(dataStep1[len(dataStep1) - 1])



i = 0
# get a n * 4 matrix from publication
with jsonlines.open('../documents/publication.jsonl') as reader:
    for item in reader:
        i = i + 1
        # if (i > 100):
        #     break
        print("current: ", i)
        doc_id = item["id"]
        print(doc_id)
        print(item["title"])
        title = getStringBasedOnStringOrList(item["title"])
        abstract = getStringBasedOnStringOrList(item["abstract"])
        topic = getStringBasedOnStringOrList(item["topic"])
        dataStep1.append([doc_id, title, abstract, topic])
        print(dataStep1[len(dataStep1) - 1])


print('good till write file')
with open('dataStep1_translate.csv', 'w') as f:
  write = csv.writer(f)
  write.writerows(dataStep1)




# save dataset as csv file 
dataStep2 = np.asarray(dataStep1)
# print(dataStep1)
print(dataStep2.shape)
# dataStep1Test = np.asarray(dataStep1Test)
np.savetxt("dataStep2.csv", dataStep2, delimiter = ',', fmt = '%s')
# np.savetxt("dataStep1Test.csv", dataStep1Test, delimiter = ',',fmt = '%s')


