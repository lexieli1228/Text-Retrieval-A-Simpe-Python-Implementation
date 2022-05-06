from threading import Thread
import socket
import json
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import data_overall_prep

data_path = './data/all_news.csv'
dictionary_path = './vocab.txt'
synonym_path = './synonym.txt'

data_load_result_csv_path = './data_modified/data_load_result_csv_path.csv'

data_tf_idf_result_csv_path = './data_modified/data_tf_idf_result_csv_path.csv'
data_tf_idf_vector_result_csv_path = './data_modified/data_tf_idf_vector_result_csv_path.csv'

data_word_conn_matrix_csv_path = './data_modified/data_word_conn_matrix_csv_path.csv'
data_keywords_matrix_csv_path = './data_modified/data_keywords_matrix_csv_path.csv'

data_tf_idf_result_csv_title_path = './data_modified/data_tf_idf_result_csv_title_path.csv'
data_tf_idf_vector_result_csv_title_path = './data_modified/data_tf_idf_vector_result_csv_title_path.csv'

data_centrality_csv_path = './data_modified/data_centrality_csv_path.csv'

first_round_num = 20
second_round_num = 20
third_round_num = 20

conn_words_num = 3

centrality_weight = 0.3
title_weight = 0.7
conn_words_weight = 0.1
keywords_weight = 0.1

class LocalServer(object):
    def __init__(self, host, port):
        self.address = (host, port)
    
    def run(self):
        
        # initialization of data, run at the first time
        # data_overall_prep.data_overall_prep()
        
        # read prepared data directly
        data_x = pd.read_csv(data_load_result_csv_path)
        print("server load data successful")

        # notice: only body version is implemented at this point
        tf_idf_result = pd.read_csv(data_tf_idf_result_csv_path)
        tf_idf_vector_result = pd.read_csv(data_tf_idf_vector_result_csv_path)
        print("server load tf-idf successful")
        
        # load synonym vocab
        synonym_vocab = []
        file_synonym = open(synonym_path, 'r')
        txt_p = file_synonym.readline()
        synonym_individuals = (txt_p[:-1]).split(' ')
        list_synonym_p = []
        for i in range(len(synonym_individuals)):
            list_synonym_p.append(synonym_individuals[i])
        synonym_vocab.append(list_synonym_p)
        while txt_p:
            txt_p = file_synonym.readline()
            synonym_individuals_temp = (txt_p[:-1]).split(' ')
            list_synonym_temp = []
            for i in range(len(synonym_individuals_temp)):
                list_synonym_temp.append(synonym_individuals_temp[i])
            synonym_vocab.append(list_synonym_temp)
        print("server load synonym vocab successful")
        
        # load key words
        keywords_vec = pd.read_csv(data_keywords_matrix_csv_path)
        print("server load keywords vectors successful")
        
        # load title tf-idf, vector
        tf_idf_result_title = pd.read_csv(data_tf_idf_result_csv_title_path)
        tf_idf_vector_result_title = pd.read_csv(data_tf_idf_vector_result_csv_title_path)
        print("server load tf-idf for title successful")
       
        # load centrality
        centrality_x = pd.read_csv(data_centrality_csv_path)
        print("server load centrality successful")
        
        # read_dictionary
        dictionary_vocab = []
        file_dictionary = open(dictionary_path, 'r')
        txt_1 = file_dictionary.readline()
        dictionary_vocab.append(txt_1[:-1])
    
        while txt_1:
            txt_1 = file_dictionary.readline()
            dictionary_vocab.append(txt_1[:-1])
       
        file_dictionary.close()
        
        print("server dictionary preparation completed")

        print("server data preparation completed")
        
        # socket preparation
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(self.address)
        server.listen(5)
        print("server socket preparation completed")

        """
        TODO: 请补充实现文本检索，以及服务器端与客户端之间的通信
        
        1. 接受客户端传递的数据， 例如检索词
        2. 调用检索函数，根据检索词完成检索
        3. 将检索结果发送给客户端，具体的数据格式可以自己定义
        
        """
        
        # task function aiming at multitask
        def task(conn, items):
            
            print("server in thread processing...")

            lemmatizer = WordNetLemmatizer()

            result = []

            if len(items) == 0:
                result = [('-1', 'please type in valid input')]
                result_json = json.dumps(result)
                conn.send(bytes(result_json.encode('utf-8')))
                print("server send result completed")
                conn.close()
            
            def rid_of_specials(text_x):
                return re.sub('[^A-Za-z]+', ' ', text_x).lower()
            
            # calculate weighted centrality
            def cal_weight_centrality(x):
                return (x - 1)*centrality_weight
            
            # calculate weighted title
            def cal_weight_title(x):
                return x*title_weight
            
            # calculate weighted synonym
            def cal_weight_conn_words(x):
                return x*conn_words_weight
            
            # calculate weighted keywords
            def cal_weight_keywords(x):
                return x*keywords_weight
            
            # indicate the number of words that is not in the vocab list
            flag_not_in_list_cnt = 0
            
            # stores target articles
            temp_dict = {}

            for index_i in range(len(items)):
            
                item_temp_0 = items[index_i]

                # get rid of special components
                item_temp_1 = rid_of_specials(item_temp_0)
                # lemmatize
                item_x = lemmatizer.lemmatize(item_temp_1)

                # basic searching
                vocab_index = -1
            
                if item_x in dictionary_vocab:
                    vocab_index = dictionary_vocab.index(item_x)
                else:
                    flag_not_in_list_cnt += 1
                    continue

                # search: version_3
                # 根据body部分tf-idf向量中的值进行排序查找，根据centrality大小和title中tf-idf的值大小加权进行排序
                # centrality: 之前计算出来的中心度
                # 鉴于本任务的文本量不大，我们并不对关键词进行初筛，而仅仅是作为一个计算排名的参考
                
                # getting key words
                keywords_result = keywords_vec.iloc[vocab_index]
                keywords_result_dict = keywords_result.to_dict()
                
                # getting articles that the target item appears
                first_round_result = tf_idf_vector_result.iloc[vocab_index]
                first_round_result_dict = first_round_result.to_dict()
                first_round_result_title = tf_idf_vector_result_title.iloc[vocab_index]

                sort_base_value = first_round_result_dict

                for i in range(1, len(sort_base_value)):
                    sort_base_value[str(i)] += cal_weight_centrality(centrality_x.iloc[i - 1][1])
                    sort_base_value[str(i)] += cal_weight_title(first_round_result_title[str(i)])
                    sort_base_value[str(i)] += cal_weight_keywords(keywords_result_dict[str(i - 1)])
                
                for i in range(1, len(sort_base_value)):
                    if str(i) in temp_dict:
                        temp_dict[str(i)] += sort_base_value[str(i)]
                    else:
                        temp_dict[str(i)] = sort_base_value[str(i)]
                
                # for every item_x, get the matrix and search for similar words
                similar_words_result = synonym_vocab[vocab_index]

                for i in range(conn_words_num):
                    item_conn = similar_words_result[i]
                    item_conn_index = dictionary_vocab.index(item_conn)
                    conn_round_result = tf_idf_vector_result.iloc[item_conn_index]
                    conn_round_result_dict = conn_round_result.to_dict()
                    conn_round_result_dict_sort = sorted(conn_round_result_dict.items(), key=lambda x:x[1], reverse = True)
                    
                    for j in range(conn_words_num):
                        if str(j) in temp_dict:
                            temp_dict[str(j)]  += cal_weight_conn_words(conn_round_result_dict_sort[i][1])
                        else:
                            temp_dict[str(j)]  = cal_weight_conn_words(conn_round_result_dict_sort[i][1])
            

            if flag_not_in_list_cnt == len(items):
                result = [('-1', 'please type in input sequence that contains at least one word in vocab list')]

            else:
                third_round_result_dict_sort = sorted(temp_dict.items(), key=lambda x:x[1], reverse = True)
                third_round_cnt = 0
                third_round_temp_result = []
                for i in range(1, len(third_round_result_dict_sort)):
                    if third_round_result_dict_sort[i][1] == 0:
                        break
                    third_round_temp_result.append(int(third_round_result_dict_sort[i][0]))
                    third_round_cnt += 1
                    if third_round_cnt >= third_round_num:
                        break
            
                for i in range(len(third_round_temp_result)):
                    temp_find_info = data_x.iloc[third_round_temp_result[i] - 1]
                    result.append((temp_find_info['title'], temp_find_info['body']))
            

            result_json = json.dumps(result)
            conn.send(bytes(result_json.encode('utf-8')))
            print("server send result completed")
            conn.close()

        # multitask
        while True:
            print("server waiting for connection...")
            conn, addr = server.accept()
            print("server connected by: {}".format(addr))
            items_recv = conn.recv(100000)
            items_json = json.loads(items_recv.decode('utf-8'))
            Thread(target=task, args=(conn, items_json)).start()

server = LocalServer("0.0.0.0", 1234)
server.run()

# below are previous search versions
                # # search: version_1
                # # 根据tf-idf向量中的值进行排序查找
                # # 在向量中相应程度越高说明越重要
                # # 直接进行排序
                # first_round_result = tf_idf_vector_result.iloc[vocab_index]
                # first_round_result_dict = first_round_result.to_dict()
                # first_round_result_dict_sort = sorted(first_round_result_dict.items(), key=lambda x:x[1], reverse = True)
            
                # # 第一轮的查找
                # first_round_cnt = 0
                # first_round_temp_result = []
                # for i in range(1, len(first_round_result_dict_sort)):
                #     if first_round_result_dict_sort[i][1] == 0:
                #         break
                #     first_round_temp_result.append(int(first_round_result_dict_sort[i][0]))
                #     first_round_cnt += 1
                #     if first_round_cnt >= first_round_num:
                #         break
            
                # for i in range(len(first_round_temp_result)):
                #     temp_find_info = data_x.iloc[first_round_temp_result[i] - 1]
                #     result.append((temp_find_info['title'], temp_find_info['body']))
            

                # # search: version_2
                # # 根据tf-idf向量中的值进行排序查找，加权centrality大小进行排序
                # # centrality: 之前计算出来的中心度
                # # 直接进行排序
                # first_round_result = tf_idf_vector_result.iloc[vocab_index]
                # first_round_result_dict = first_round_result.to_dict()

                # sort_base_value = first_round_result_dict
            
                # second_round_result_dict_sort = sorted(sort_base_value.items(), key=lambda x:x[1], reverse = True)
                # second_round_cnt = 0
                # second_round_temp_result = []
                # for i in range(1, len(second_round_result_dict_sort)):
                #     if second_round_result_dict_sort[i][1] == 0:
                #         break
                #     second_round_temp_result.append(int(second_round_result_dict_sort[i][0]))
                #     second_round_cnt += 1
                #     if second_round_cnt >= second_round_num:
                #         break
            
                # for i in range(len(second_round_temp_result)):
                #     temp_find_info = data_x.iloc[second_round_temp_result[i] - 1]
                #     result.append((temp_find_info['title'], temp_find_info['body']))