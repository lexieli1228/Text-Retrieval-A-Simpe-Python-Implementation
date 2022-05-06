import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import networkx as nx
from sklearn.cluster import KMeans, MeanShift
from sklearn import datasets, preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics

import data_loader
import data_preparation
import data_graph_building

data_path = './data/all_news.csv'
dictionary_path = './vocab.txt'

data_load_result_csv_path = './data_modified/data_load_result_csv_path.csv'
data_tf_idf_result_csv_path = './data_modified/data_tf_idf_result_csv_path.csv'
data_tf_idf_vector_result_csv_path = './data_modified/data_tf_idf_vector_result_csv_path.csv'
data_centrality_csv_path = './data_modified/data_centrality_csv_path.csv'
cluster_verification_result_path = './verification_results/cluster_verification.txt'
cluster_verification_result_path_with_title = './verification_results/cluster_verification_with_title.txt'

threshold_x = 0.2

# notice: this function is implemented under the guidance of online resources
def calculate_purity(label_annotation, label_prediction):
    temp_matrix = metrics.cluster.contingency_matrix(label_annotation, label_prediction)
    return np.sum(np.amax(temp_matrix, axis=0))/np.sum(temp_matrix)


# verification function
def cluster_verification():
    # preparing data
    data_x = data_loader.load_data()
    print("cluster verification load data completed")

    tf_idf_result, tf_idf_vector_result = data_preparation.set_tf_idf(data_x)
    print("cluster verification calculate tf-idf completed")

    vector_result = np.array(pd.DataFrame(tf_idf_vector_result)).T
    print("cluster verification getting original vector completed")

    pca_vector_result = data_graph_building.pca_vector_modification(vector_result) 
    print("cluster verification getting pca vector completed")

    tf_idf_result_title, tf_idf_vector_result_title = data_preparation.set_tf_idf_title(data_x)
    print("cluster verification calculate tf-idf title completed")

    vector_result_title = np.array(pd.DataFrame(tf_idf_vector_result_title)).T
    print("cluster verification getting original title vector completed")

    pca_vector_result_title = data_graph_building.pca_vector_modification(vector_result_title) 
    print("cluster verification getting title pca vector completed")
    
    # getting cluster number
    gt_cluster_number = data_x.topic.nunique()

    # original vector
    km_vec = KMeans(n_clusters=gt_cluster_number)
    km_vec.fit(vector_result)
    label_vec = km_vec.labels_
    
    # pca vector result
    vector_result_pca = data_graph_building.pca_vector_modification(vector_result)
    km_pca = KMeans(n_clusters=gt_cluster_number)
    km_pca.fit(vector_result_pca)
    label_pca = km_pca.labels_

    # original title vector
    km_vec_title = KMeans(n_clusters=gt_cluster_number)
    km_vec_title.fit(vector_result_title)
    label_vec_title = km_vec_title.labels_
    
    # pca vector result
    vector_result_pca_title = data_graph_building.pca_vector_modification(vector_result_title)
    km_pca_title = KMeans(n_clusters=gt_cluster_number)
    km_pca_title.fit(vector_result_pca_title)
    label_pca_title = km_pca_title.labels_
    
    # getting original label
    label_original = list(data_x['topic'])
    
    # getting purity results
    result_original = calculate_purity(label_original, label_vec)
    result_pca = calculate_purity(label_original, label_pca)
    result_original_title = calculate_purity(label_original, label_vec_title)
    result_pca_title = calculate_purity(label_original, label_pca_title)
    
    # write to file
    file_verification_result = open(cluster_verification_result_path_with_title, 'w')

    string_1 = 'the purity result of original vector is: ' + str(result_original)
    string_2 = 'the purity result of pca vector is: ' + str(result_pca)
    string_3 = 'the purity result of original title vector is: ' + str(result_original_title)
    string_4 = 'the purity result of title pca vector is: ' + str(result_pca_title)
    
    file_verification_result.write(string_1 + '\n')
    file_verification_result.write(string_2 + '\n')
    file_verification_result.write(string_3 + '\n')
    file_verification_result.write(string_4 + '\n')

    print("cluster verification completed")


cluster_verification()