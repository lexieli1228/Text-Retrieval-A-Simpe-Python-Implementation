import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

data_path = './data/all_news.csv'
dictionary_path = './vocab.txt'

# 原数据读入后的基础处理
def load_data():
    
    # 直接读入
    data = pd.read_csv(data_path)
    
    # 去掉特殊的字符
    def rid_of_specials(text_x):
        return re.sub('[^A-Za-z]+', ' ', text_x).lower()
    
    # 对body和title进行分别处理
    data["body"] = data["body"].astype(str).apply(rid_of_specials)
    data["title"] = data["title"].astype(str).apply(rid_of_specials)
    
    # 去除stopwords
    sw_nltk = (stopwords.words('english'))
    stop_words = set(sw_nltk)

    def remove_sw(text_x):
        text_x = text_x.split(' ')
        return  ' '.join(z for z in text_x if z not in stop_words)
    
    stopped_body = data["body"].apply(remove_sw)
    stopped_title = data["title"].apply(remove_sw)
    
    # 标准化
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
    
    # 将处理过的文本存入data
    data["prepared body"] = lemmatized_body
    data["prepared title"] = lemmatized_title
    
    # 制作字典
    vocab_set = set()

    def add_words(text_x):
        for i in text_x:
            temp_list = i.split(' ')
            for j in temp_list:
                if j != '':
                    vocab_set.add(j)
    
    add_words(lemmatized_body)
    add_words(lemmatized_title)
    
    # 形成字典并排序
    result = list(vocab_set)
    result.sort()
    
    # 写入
    file = open(dictionary_path, 'w')
    
    for i in result:
        file.write(str(i) + '\n')
    
    file.close()

    return data