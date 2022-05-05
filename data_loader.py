import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

data_path = './data/all_news.csv'
dictionary_path = './vocab.txt'


def load_data():
   
    data = pd.read_csv(data_path)

    def rid_of_specials(text_x):
        return re.sub('[^A-Za-z]+', ' ', text_x).lower()

    data["body"] = data["body"].astype(str).apply(rid_of_specials)
    data["title"] = data["title"].astype(str).apply(rid_of_specials)

    sw_nltk = (stopwords.words('english'))
    stop_words = set(sw_nltk)

    def remove_sw(text_x):
        text_x = text_x.split(' ')
        return  ' '.join(z for z in text_x if z not in stop_words)
    
    stopped_body = data["body"].apply(remove_sw)
    stopped_title = data["title"].apply(remove_sw)
    
    lemmatizer = WordNetLemmatizer()
    
    lemmatized_body = []
    for i in stopped_body:
        sentence_temp = i.split()
        sentence_temp_modified = []
        for j in sentence_temp:
            sentence_temp_modified.append(lemmatizer.lemmatize(j))
        lemmatized_body.append((' '.join(z for z in sentence_temp_modified)))

    lemmatized_title = []
    for i in stopped_title:
        sentence_temp = i.split()
        sentence_temp_modified = []
        for j in sentence_temp:
            sentence_temp_modified.append(lemmatizer.lemmatize(j))
        lemmatized_title.append((' '.join(z for z in sentence_temp_modified)))  
    

    data["prepared body"] = lemmatized_body
    data["prepared title"] = lemmatized_title

    vocab_set = set()

    def add_words(text_x):
        for i in text_x:
            temp_list = i.split(' ')
            for j in temp_list:
                if j != '':
                    vocab_set.add(j)
    
    add_words(lemmatized_body)
    add_words(lemmatized_title)

    result = list(vocab_set)
    result.sort()

    file = open(dictionary_path, 'w')
    
    for i in result:
        file.write(str(i) + '\n')
    
    file.close()

    return data