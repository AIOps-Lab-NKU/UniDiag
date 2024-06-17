import numpy as np
from scipy.stats import wasserstein_distance
from scipy.stats import pearsonr
from scipy import stats
from numpy import *
import math
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import pandas as pd
import os
from Cluster.cluster_stat import find_max,cluster_stat
from Cluster.score import *



def cos_sim(vector_a, vector_b):
    """
    calculate the cosine similarity between two vectors
    :param vector_a: vector a 
    :param vector_b: vector b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    sim = num / denom
    return sim

def euclidean_distance(x1,x2,p=2):
    return sum(abs(x1 - x2) ** p) ** (1./p)

def manhatton_distance(x1,x2,p=1):
    return sum(abs(x1 - x2) ** p) ** (1./p)
        
def qiebixuefu_distance(x1,x2):
    return np.linalg.norm(x1-x2,ord=np.inf)

def cos_distance(x1,x2):
    a = sum(x1*x2)
    b = sum(x1 ** 2)**(0.5) * sum(x2 ** 2)**(0.5)
    return 1-a/b

def Square(x, y):
    return np.sum(np.power((x - y), 2))

def Wasserstein(P,Q):
    dists=[i for i in range(len(P))]
    return stats.wasserstein_distance(P,Q)

def calcSBDncc(x,y,s):
    assert len(x)==len(y)
    assert isinstance(s,int)
    length_ = len(x)
    pow_x = 0
    pow_y = 0
    ccs = 0
    for i in range(length_-s):
        ccs +=  x[i+s]*y[i]
        pow_x += math.pow(x[i+s],2)
        pow_y += math.pow(y[i],2)
    dist_x =math.pow(pow_x,0.5)
    dist_y =math.pow(pow_y,0.5)
    dist_xy = dist_x*dist_y
    ncc = ccs/dist_xy
    return ncc
def calcSBD(x,y,s=None):
    assert len(x)==len(y)
    if  s==None:
        length_ = len(x)
        ncc_list = []
        for i in range(int(length_*0.5)):  # 0.5: ensures that at least half of the data is used for the correlation calculation
            ncc_list.append(calcSBDncc(x,y,i))
        ncc = max(ncc_list)
        delay = ncc_list.index(max(ncc_list))
        sbd = 1 - ncc
    else:
        ncc = calcSBDncc(x,y,s)
        delay = s  # Passed as alarm delay number.
        sbd = 1 - ncc #sbd: data is 0-2, so it is more appropriate to return ncc for explaining the correlation here.
    return sbd

def cluster_main(config,if_turn = False):
    cluster_config = config['cluster']
    input_data_path = cluster_config['input_data_path']
    a = np.load(input_data_path)
    all_data=a['x']
    label = a['y']

    train_data = []
    test_data = []
    train_label = []
    test_label = []
    index = [0,1,3,4,5,6,9,13,14,16,17,18,20,23,24,25,26,27,28,30,31,32,33,35,36,39,40,43,44,46,
    47,48,50,51,53,54,56,58,59,60,62,63,64,65,67,68,70,71,72,74,76,77,79,80,81,82,83,84,85,
    86,90,91,92,93,95,98,99,100,101,102,104,105,106,107,108,109,110,113,114,115,116,
    117,
    118,
    119,
    120,
    121,
    124,
    125]
    # train_index: all time index of cases in the training set.
    train_index = []
    for i in range(len(index)):
        if index[i]<49:
            temp = [j for j in range(index[i]*10,index[i]*10+10)]
            train_index += temp
        else:
            temp = [j for j in range(index[i]*10-1,index[i]*10+9)]
            train_index += temp

    for i in range(len(all_data)):
        if i in train_index:
            train_data.append(all_data[i])
            train_label.append(label[i])
        else:
            test_data.append(all_data[i])
            test_label.append(label[i])
    model_1 = AgglomerativeClustering(n_clusters=2).fit(train_data)
    agg_labels_1 = model_1.labels_
    # print(agg_labels_1)
    # 加条件判断
    if if_turn:
        for i in range(len(agg_labels_1)):
            if agg_labels_1[i] == 0:
                agg_labels_1[i] =1
            else:
                agg_labels_1[i] =0

    n_cluster = 2
    n_vector = 100 # embedding vectors' dimension
    num = [0 for i in range(n_cluster)]
    cluster_center_1 = [np.array([0.0 for i in range(n_vector)]) for j in range(n_cluster)]
    for i in range(len(train_data)):
        cluster_center_1[agg_labels_1[i]] += train_data[i]
        num[agg_labels_1[i]] += 1
    for i in range(n_cluster):
        cluster_center_1[i] /= num[i]
    cluster_center_type_1 = [0,1]

    test_pred_label_1 = []
    for i in range(len(test_data)):
        temp_center_index = 0
        temp_center_dist = 100000
        for j in range(len(cluster_center_1)):
            if euclidean_distance(test_data[i],cluster_center_1[j]) < temp_center_dist:
                temp_center_dist = euclidean_distance(test_data[i],cluster_center_1[j])
                temp_center_index = j
        test_pred_label_1.append(cluster_center_type_1[temp_center_index])

    anomaly_train_data = []
    anomaly_train_label = []
    anomaly_test_data = []
    anomaly_test_label = []
    normal_test_label = []
    for i in range(len(agg_labels_1)):
        if agg_labels_1[i] == 1:
            anomaly_train_data.append(train_data[i])
            anomaly_train_label.append(train_label[i])

    for i in range(len(test_pred_label_1)):
        if test_pred_label_1[i] == 1:
            anomaly_test_data.append(test_data[i])
            anomaly_test_label.append(test_label[i])
        else:
            normal_test_label.append(test_pred_label_1[i])
    
    result = pd.DataFrame()
    result['label']= test_label
    # Failure type Diagnosis
    n_cluster = 15 # cluster numbers
    # agg_model=KMeans(n_clusters=n_cluster, random_state=0).fit(anomaly_train_data)
    agg_model = AgglomerativeClustering(n_clusters=n_cluster).fit(anomaly_train_data)
    # agg_model=DBSCAN(eps=0.07,min_samples=20).fit(anomaly_train_data)#eps is Radius.
    agg_labels = agg_model.labels_
    num = [0 for i in range(n_cluster)]
    true_cluster_center = [np.array([0.0 for i in range(n_vector)]) for j in range(n_cluster)]
    for i in range(len(anomaly_train_data)):
        true_cluster_center[agg_labels[i]]+=anomaly_train_data[i]
        num[agg_labels[i]]+=1
    for i in range(n_cluster):
        true_cluster_center[i] /= num[i]
        
    dis = [100000 for i in range(n_cluster)] 
    cluster_center = [np.array([0.0 for i in range(n_vector)]) for j in range(n_cluster)]
    cluster_center_type = [-1 for i in range(n_cluster)]
    for i in range(len(anomaly_train_data)):
        if euclidean_distance(anomaly_train_data[i],true_cluster_center[agg_labels[i]]) < dis[agg_labels[i]]:
            dis[agg_labels[i]]=euclidean_distance(anomaly_train_data[i],true_cluster_center[agg_labels[i]])
            cluster_center[agg_labels[i]]=anomaly_train_data[i]
            cluster_center_type[agg_labels[i]]=anomaly_train_label[i]

    test_pred_label = []
    for i in range(len(anomaly_test_data)):
        temp_center_index = 0
        temp_center_dist = 100000
        for j in range(len(cluster_center)):
            if euclidean_distance(anomaly_test_data[i],cluster_center[j]) < temp_center_dist:
                temp_center_dist = euclidean_distance(anomaly_test_data[i],cluster_center[j])
                temp_center_index = j
        test_pred_label.append(cluster_center_type[temp_center_index])

    pred = []
    j=0
    for i in range(len(test_pred_label_1)):
        if test_pred_label_1[i]==1:
            pred.append(test_pred_label[j])
            j+=1
        else:
            pred.append(test_pred_label_1[i])
   
    result['result'] = pred
    if not os.path.exists(os.path.split(cluster_config["output_data_path"])[0]):
        os.mkdir(os.path.split(cluster_config["output_data_path"])[0])
    result.to_csv(cluster_config["output_data_path"],index=False)

    # 10 snapshots to 1 result:
    cluster_stat(cluster_config['output_data_path'],cluster_config['stat_output_path'],config)
    
    # score:
    score_report(cluster_config['stat_output_path'],cluster_config['report_output_path'])

if __name__ == '__main__':
    cluster_main(None)
