import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import data_graph_building

data_path = './data/all_news.csv'
dictionary_path = './vocab.txt'
synonym_path = './synonym.txt'

key_words_num = 6
synonym_num = 10


def get_tf_body(x):
    t = dict(x['word body'].value_counts()/x['word body'].value_counts().sum())
    return t


def get_idf_body(x, total_num):
    return np.log(total_num/x['id'].nunique())


def form_vec_body(x, word_list):
    vec_x = np.zeros(len(word_list))
    for i in range(x.shape[0]):
        if x['word body'].iloc[i] in word_list:
            vec_x[word_list.index(x['word body'].iloc[i])] = x['TF-IDF_body'].iloc[i]
    return vec_x


def set_tf_idf(data_x):
    # new dataframe
    split_words_body = data_x["prepared body"].str.split(' ', expand=True).stack().rename('word body').reset_index()
    new_data_body = pd.merge(data_x['id'], split_words_body, left_index=True, right_on='level_0')
    
    # set tf
    dicts = dict(new_data_body.groupby("id").apply(get_tf_body))
    new_data_body['TF_body'] = new_data_body.apply(lambda x: dicts[x["id"]][x['word body']], axis=1)
    
    # set idf
    total_num_body = data_x["id"].nunique()
    word_cluster_body = dict(new_data_body.groupby('word body').apply(lambda x: get_idf_body(x, total_num_body)))
    new_data_body['IDF_body'] = new_data_body.apply(lambda x: word_cluster_body[x['word body']], axis=1)
    
    # set tf-idf
    new_data_body['TF-IDF_body'] = new_data_body['TF_body']*new_data_body['IDF_body']
    
    # calculate vector
    dictionary_vocab = []
    file_dictionary = open(dictionary_path, 'r')
    txt_1 = file_dictionary.readline()
    dictionary_vocab.append(txt_1[:-1])
    
    while txt_1:
        txt_1 = file_dictionary.readline()
        dictionary_vocab.append(txt_1[:-1])

    vector_data_body = dict(new_data_body.groupby('id').apply(lambda x: form_vec_body(x, dictionary_vocab)))

    return new_data_body, vector_data_body


def get_tf_title(x):
    t = dict(x['word title'].value_counts()/x['word title'].value_counts().sum())
    return t


def get_idf_title(x, total_num):
    return np.log(total_num/x['id'].nunique())


def form_vec_title(x, word_list):
    vec_x = np.zeros(len(word_list))
    for i in range(x.shape[0]):
        if x['word title'].iloc[i] in word_list:
            vec_x[word_list.index(x['word title'].iloc[i])] = x['TF-IDF_title'].iloc[i]
    return vec_x


def set_tf_idf_title(data_x):
    # new dataframe
    split_words_title = data_x["prepared title"].str.split(' ', expand=True).stack().rename('word title').reset_index()
    new_data_title = pd.merge(data_x['id'], split_words_title, left_index=True, right_on='level_0')
    
    # set tf
    dicts = dict(new_data_title.groupby("id").apply(get_tf_title))
    new_data_title['TF_title'] = new_data_title.apply(lambda x: dicts[x["id"]][x['word title']], axis=1)
    
    # set idf
    total_num_title = data_x["id"].nunique()
    word_cluster_title = dict(new_data_title.groupby('word title').apply(lambda x: get_idf_title(x, total_num_title)))
    new_data_title['IDF_title'] = new_data_title.apply(lambda x: word_cluster_title[x['word title']], axis=1)
    
    # set tf-idf
    new_data_title['TF-IDF_title'] = new_data_title['TF_title']*new_data_title['IDF_title']
    
    # calculate vector
    dictionary_vocab = []
    file_dictionary = open(dictionary_path, 'r')
    txt_1 = file_dictionary.readline()
    dictionary_vocab.append(txt_1[:-1])
    
    while txt_1:
        txt_1 = file_dictionary.readline()
        dictionary_vocab.append(txt_1[:-1])

    vector_data_title = dict(new_data_title.groupby('id').apply(lambda x: form_vec_title(x, dictionary_vocab)))

    return new_data_title, vector_data_title


def cal_words_synonym(tf_idf_vector_result, dictionary_vocab):
    words_vec = np.array(pd.DataFrame(tf_idf_vector_result))
    words_vec_result = data_graph_building.pca_vector_modification(words_vec)
    words_similarity = (data_graph_building.cal_cosine_similarity(words_vec_result))
    words_similarity_df = (pd.DataFrame(words_similarity))
    file_synonym = open(synonym_path, 'w')
    for i in range(len(words_similarity)):
        temp = words_similarity_df.iloc[i]
        temp_similarity_dict = temp.to_dict()
        temp_similarity_dict_sort = sorted(temp_similarity_dict.items(), key=lambda x:x[1], reverse = True)
        list_i = []
        string_i = dictionary_vocab[i]
        for j in range(synonym_num + 1):
            if dictionary_vocab[temp_similarity_dict_sort[j][0]] == dictionary_vocab[i]:
                continue
            list_i.append(dictionary_vocab[temp_similarity_dict_sort[j][0]])
        
        for k in range(len(list_i)):
            string_i += ' '
            string_i += list_i[k]
    
        string_i += '\n'
        file_synonym.write(string_i)
    
    file_synonym.close()


def cal_keywords_vec(tf_idf_vector_result, dictionary_vocab, data_x):
    keywords_vec = np.zeros((len(data_x), len(dictionary_vocab)))
    for i in range(len(data_x)):
        temp_vec = tf_idf_vector_result.iloc[i]
        temp_vec_dict = temp_vec.to_dict()
        temp_vec_dict_sort = sorted(temp_vec_dict.items(), key=lambda x:x[1], reverse = True)
        for j in range(0, key_words_num + 1):
            if temp_vec_dict_sort[j][0][0] != 'U':
                keywords_vec[i][int(temp_vec_dict_sort[j][0])] = 1
    return keywords_vec