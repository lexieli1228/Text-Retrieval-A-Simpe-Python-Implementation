from cv2 import threshold
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import networkx as nx

import data_loader
import data_preparation
import data_graph_building

data_path = './data/all_news.csv'
dictionary_path = './vocab.txt'

data_load_result_csv_path = './data_modified/data_load_result_csv_path.csv'

data_tf_idf_result_csv_path = './data_modified/data_tf_idf_result_csv_path.csv'
data_tf_idf_vector_result_csv_path = './data_modified/data_tf_idf_vector_result_csv_path.csv'

data_word_conn_matrix_csv_path = './data_modified/data_word_conn_matrix_csv_path.csv'
data_keywords_matrix_csv_path = './data_modified/data_keywords_matrix_csv_path.csv'

data_tf_idf_result_csv_title_path = './data_modified/data_tf_idf_result_csv_title_path.csv'
data_tf_idf_vector_result_csv_title_path = './data_modified/data_tf_idf_vector_result_csv_title_path.csv'

data_centrality_csv_path = './data_modified/data_centrality_csv_path.csv'

threshold_x = 0.2

# data overall preparation
# run at the first time
def data_overall_prep():
    
    print("server beginning data overall prepping")

    data_x = data_loader.load_data()
    data_x.to_csv(data_load_result_csv_path)
    print("server save data successful")
    
    dictionary_vocab = []
    file_dictionary = open(dictionary_path, 'r')
    txt_1 = file_dictionary.readline()
    dictionary_vocab.append(txt_1[:-1])
    while txt_1:
        txt_1 = file_dictionary.readline()
        dictionary_vocab.append(txt_1[:-1])
    print("server read vocabulary successful")

    # tf-idf for body part
    tf_idf_result, tf_idf_vector_result = data_preparation.set_tf_idf(data_x)
    tf_idf_result_save = pd.DataFrame(tf_idf_result)
    tf_idf_result_save.to_csv(data_tf_idf_result_csv_path)
    tf_idf_vector_result_save = pd.DataFrame(tf_idf_vector_result)
    tf_idf_vector_result_save.to_csv(data_tf_idf_vector_result_csv_path)
    print("server save tf-idf successful")

    # saving word conn matrix
    data_preparation.cal_words_synonym(tf_idf_vector_result, dictionary_vocab)
    print("server save synonym successful")

    # saving keywords matrix
    keywords_vec = data_preparation.cal_keywords_vec(tf_idf_vector_result, dictionary_vocab, data_x)
    keywords_vec_save = pd.DataFrame(keywords_vec.T)
    keywords_vec_save.to_csv(data_keywords_matrix_csv_path)
    print("server save keywords vectors successful")

    # tf-idf for title part
    tf_idf_result_title, tf_idf_vector_result_title = data_preparation.set_tf_idf_title(data_x)
    tf_idf_result_save_title = pd.DataFrame(tf_idf_result_title)
    tf_idf_result_save_title.to_csv(data_tf_idf_result_csv_title_path)
    tf_idf_vector_result_save_title = pd.DataFrame(tf_idf_vector_result_title)
    tf_idf_vector_result_save_title.to_csv(data_tf_idf_vector_result_csv_title_path)
    print("server save tf-idf for title successful")
    
    # calculate centrality
    centrality_x = data_graph_building.cal_centrality_x(tf_idf_vector_result, threshold_x)
    centrality_x_store = pd.DataFrame(centrality_x)
    centrality_x_store.to_csv(data_centrality_csv_path)
    print("server save centrality successful")

    print("server data overall prepping completed")