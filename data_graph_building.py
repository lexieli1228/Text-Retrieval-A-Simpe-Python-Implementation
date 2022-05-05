import numpy as np
import pandas as pd
import re
import nltk
import networkx as nx
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data_path = './data/all_news.csv'
dictionary_path = './vocab.txt'

n_components = 1000
n_times = 10000

threshold_centrality = 2

def pca_vector_modification(original_vectors):
    vector_result_temp = original_vectors.copy()
    # X_std = StandardScaler().fit_transform(original_vectors)

    for i in range(len(original_vectors)):
        for j in range(len(original_vectors[0])):
            if original_vectors[i][j] != 0:
                vector_result_temp[i][j] = original_vectors[i][j]*n_times

    pca = PCA(n_components = n_components)
    result_vector = pca.fit_transform(vector_result_temp)

    return result_vector


def cal_cosine_similarity(x):
    matrix_x = []
    for i in range(x.shape[0]):
        matrix_x.append(x[i]/np.sqrt(np.sum(np.square(x[i]))))
    return np.array(matrix_x) @ (np.array(matrix_x).T)


def set_threshold_x(similarity_matrix, threshold_x):
    similarity_matrix_new = np.zeros((similarity_matrix.shape[0], similarity_matrix.shape[0]))

    for i in range(similarity_matrix.shape[0]):
        for j in range(similarity_matrix.shape[1]):
            similarity_matrix_new[i][j] = 2 - similarity_matrix[i][j] 
            if abs(similarity_matrix_new[i][j]) < threshold_x:
                similarity_matrix_new[i][j] = 0

    return similarity_matrix_new


def cal_centrality_x(tf_idf_vector_result, threshold_x):

    vector_result = np.array(pd.DataFrame(tf_idf_vector_result)).T
    pca_vector_result = pca_vector_modification(vector_result)

    # calculate similarity matrix
    similarity_matrix = cal_cosine_similarity(pca_vector_result)

    # set threshold
    similarity_matrix_modified = set_threshold_x(similarity_matrix, threshold_x)
    
    # calculate centrality
    centrality_x = np.zeros(similarity_matrix_modified.shape[0])
    for i in range(similarity_matrix_modified.shape[0]):
        temp = 0
        for j in range(similarity_matrix_modified.shape[1]):
            if similarity_matrix_modified[i][j] > threshold_centrality:
                temp += 1
        if temp == 0:
            centrality_x[i] = 0
        else:
            centrality_x[i] = similarity_matrix_modified.shape[0] / temp
    
    return centrality_x