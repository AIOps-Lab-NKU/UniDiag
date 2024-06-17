import os
import numpy as np
import pandas as pd
import shutil
import time


def GAIA_anomaly_case(data_input,last_timestamp,anomaly_name,logger):
    anomalies_case_path = os.path.join(data_input,anomaly_name)

    # test set：
    anomalies_case_list = sorted(os.listdir(anomalies_case_path))
    for i in range(len(anomalies_case_list)):
        path_1 = os.path.join(anomalies_case_path,anomalies_case_list[i])
        test_path = os.path.join(path_1,'test.txt')
        test_df = pd.read_csv(test_path,header=None,sep='\t')
        test_df.columns = ['subjectID','rel','objectID','timestamp','nan']

        # remove last column
        test_df = test_df.drop(columns=['nan'])
        # test set period: 20 min before failure and 10 min after failure.
        test_timestamp = sorted(list(set(test_df['timestamp'].tolist())))
        if len(test_timestamp)<30:
            print('{}:{}',anomalies_case_list[i],len(test_timestamp),file=logger)
            break

        # only pick period in failure. Change original timestamp for training
        test_df = test_df[test_df['timestamp'].isin(test_timestamp[20:30])]
        test_timestamp = sorted(list(set(test_df['timestamp'].tolist())))
        # new timestamp follows last timestamp in training set.
        test_df['new_timestamp'] = test_df['timestamp'].apply((lambda x: test_timestamp.index(x)+last_timestamp+1))
        
        test_df = test_df.sort_values(by=['new_timestamp'])
        new_timestamp = sorted(list(set(test_df['new_timestamp'].tolist())))

        # output. Only select several columns within.
        test_df.to_csv(os.path.join(path_1,'test.txt'),columns=['subjectID','rel','objectID','new_timestamp'],header=None,index=False,sep='\t')
        print("{} anomaly finish ========√".format(anomalies_case_list[i]),file=logger)
        print("anomaly last timestamp is {}".format(new_timestamp[-1]),file=logger)


# copy normal training set and test set.
# Attention: Name problem in GAIA dataset. There is ' '.
def cp_train(case_input,anomaly_name,logger):
    normal_case_path = os.path.join(case_input,'normal')
    anomalies_case_path = os.path.join(case_input,anomaly_name)
    anomalies_case_list = os.listdir(anomalies_case_path)


    train_path = os.path.join(normal_case_path,'train.txt')
    train_df = pd.read_csv(train_path,header=None,sep='\t')
    
    valid_path = os.path.join(normal_case_path,'valid.txt')
    valid_df = pd.read_csv(valid_path,header=None,sep='\t')

    # access_permission_denied should be specially processed.
    if anomaly_name == 'access_permission_denied':

        entity_path = os.path.join(normal_case_path,'entity2id.txt')
        entity_df = pd.read_csv(entity_path,header=None,sep='\t')

        relation_path = os.path.join(normal_case_path,'relation2id.txt')
        relation_df = pd.read_csv(relation_path,header=None,sep='\t')

        stat_path = os.path.join(normal_case_path,'stat.txt')
        stat_df = pd.read_csv(stat_path,header=None,sep='\t')

        for i in range(len(anomalies_case_list)):
            path_t = os.path.join(anomalies_case_path,anomalies_case_list[i])
            train_df.to_csv(os.path.join(path_t,'train.txt'),header=False,index=False,sep='\t')
            valid_df.to_csv(os.path.join(path_t,'valid.txt'),header=False,index=False,sep='\t')
            entity_df.to_csv(os.path.join(path_t,'entity2id.txt'),header=False,index=False,sep='\t')
            relation_df.to_csv(os.path.join(path_t,'relation2id.txt'),header=False,index=False,sep='\t')
            stat_df.to_csv(os.path.join(path_t,'stat.txt'),header=False,index=False,sep='\t')
            print("{} copy finish======√".format(anomalies_case_list[i]),file=logger)
    else:
        for i in range(len(anomalies_case_list)):
            path_t = os.path.join(anomalies_case_path,anomalies_case_list[i])
            train_df.to_csv(os.path.join(path_t,'train.txt'),header=False,index=False,sep='\t')
            valid_df.to_csv(os.path.join(path_t,'valid.txt'),header=False,index=False,sep='\t')
            print("{} copy finish======√".format(anomalies_case_list[i]),file=logger)


# process access_permission_denied:
# test contain 20min before failure, 60min in failure and 10min after failure.
# remove before and after, and split 60 into 10*6.
def for_access_permission_denied(data_input,last_timestamp,anomaly_name,logger):
    anomalies_case_path = os.path.join(data_input,anomaly_name)

    # test
    anomalies_case_list = sorted(os.listdir(anomalies_case_path))
    print(len(anomalies_case_list),file=logger)
    for i in range(len(anomalies_case_list)):
        path_1 = os.path.join(anomalies_case_path,anomalies_case_list[i])
        test_path = os.path.join(path_1,'test.txt')
        test_df = pd.read_csv(test_path,header=None,sep='\t')
        test_df.columns = ['subjectID','rel','objectID','timestamp','nan']

        # remove last column
        test_df = test_df.drop(columns=['nan'])
        # test set timestamp
        test_timestamp = sorted(list(set(test_df['timestamp'].tolist())))
        if len(test_timestamp)<80:
            print('{}:{}',anomalies_case_list[i],len(test_timestamp),file=logger)
            break

        # in failure: change original timestamp for training.
        for j in range(6):
            test_df_s = test_df[test_df['timestamp'].isin(test_timestamp[20+10*j:20+10*(j+1)])].copy()
            test_timestamp_s = sorted(list(set(test_df_s['timestamp'].tolist())))
            # new timestamp follows last timestamp in training set.
            test_df_s['new_timestamp'] = test_df_s['timestamp'].apply((lambda x: test_timestamp_s.index(x)+last_timestamp+1))
            
            test_df_s = test_df_s.sort_values(by=['new_timestamp'])
            new_timestamp = sorted(list(set(test_df_s['new_timestamp'].tolist())))

            # output: only select several columns
            # mkdir 
            path_2 = os.path.join(anomalies_case_path,'{}_{}'.format(anomalies_case_list[i],j))
            if not os.path.exists(path_2):
                os.mkdir(path_2)

            test_df_s.to_csv(os.path.join(path_2,'test.txt'),columns=['subjectID','rel','objectID','new_timestamp'],header=None,index=False,sep='\t')
            print("{} anomaly {} finish ========√".format(anomalies_case_list[i],j),file=logger)
            print("anomaly last timestamp is {}".format(new_timestamp[-1]),file=logger)

        del test_df
        # remove original dir
        shutil.rmtree(path_1)
        print("{} anomaly finish ========√".format(anomalies_case_list[i]),file=logger)

# process normal case:
# 1. choose training set last 20min to last 10 min as validtaion set, last 10 min as test set.
# 2. timestamps in test set in failure should follow the last timestamp in normal set.
def normal_case(data_input,logger):
    normal_case_path = os.path.join(data_input,'normal')

    train_path = os.path.join(normal_case_path,'train.txt')
    train_df = pd.read_csv(train_path,header=None,sep='\t')
    train_df.columns = ['subjectID','rel','objectID','timestamp','nan']
    
    # remove last column
    train_df = train_df.drop(columns=['nan'])
    # sort by timestamp
    train_df =  train_df.sort_values(by=['timestamp'])
    # record training set timestamp
    train_timestamp = sorted(list(set(train_df['timestamp'].tolist())))
    # rewrite
    train_df.to_csv(train_path,header=False,index=False,sep='\t')

    print("train finish========√",file=logger)
    print("last timestamp is {}".format(train_timestamp[-1]),file=logger)

    # choose last 20min to last 10min as validation.
    valid_df = train_df[train_df['timestamp'].isin(train_timestamp[-20:-10])] 
    valid_df = valid_df.sort_values(by=['timestamp'])
    test_df = train_df[train_df['timestamp'].isin(train_timestamp[-10:])] 
    test_df = test_df.sort_values(by=['timestamp'])
    
    # output to valid and test
    valid_df.to_csv(os.path.join(normal_case_path,'valid.txt'),header=None,index=False,sep='\t')
    test_df.to_csv(os.path.join(normal_case_path,'test.txt'),header=None,index=False,sep='\t')
        
    print("valid and test finish =======√",file=logger)

    return train_timestamp[-1]


def get_train_lasttime(case_input):
    normal_case_path = os.path.join(case_input,'normal')
    train_path = os.path.join(normal_case_path,'train.txt')
    # get last time of normal case:
    train_df = pd.read_csv(train_path,header=None,sep='\t')
    train_df.columns = ['subjectID','rel','objectID','timestamp']
    # sort by timestamp
    train_df =  train_df.sort_values(by=['timestamp'])
    # record timestamp
    train_timestamp = sorted(list(set(train_df['timestamp'].tolist())))
    return train_timestamp[-1]


# main process for GAIA
def for_GAIA_process(data_input,logfile):
    # logger
    if not os.path.exists('EntityEmbedding/log'):
        os.makedirs('EntityEmbedding/log')
    logger = open(logfile,'w')
    anomalies_list = os.listdir(data_input)
    last_timestamp = normal_case(data_input,logger)
    
    # last_timestamp = get_train_lasttime(data_input)
    for i in anomalies_list:
        # skip normal
        if i=='normal':
            continue
        # access_permission_denied 
        if i == 'access_permission_denied':
            for_access_permission_denied(data_input,last_timestamp,i,logger) 
        else:
            GAIA_anomaly_case(data_input,last_timestamp,i,logger)

        # cp 
        cp_train(data_input,i,logger)
    logger.close()

