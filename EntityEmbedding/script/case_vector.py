import numpy as np
import pandas as pd
import copy
import os

# sort case dataset by timestamp
def sort_by_timestamp(anomalies_case_list):
    dict = {}
    for i in anomalies_case_list:
        dict[i] = i[-10:]
    dict = sorted(dict.items(),key=lambda x:x[1])
    new_list = [i[0] for i in dict]
    return new_list
    

def for_GAIA_vector(config):
    """
    GAIA Dir structure
    GAIA
        - normal
        - access_permission_denied
        - file_moving
        - memory_anomalies 
        - normal_memory_freed_label
    """

    case_name = config['failure_type']
    embedding_path = config['output_data_path']
    normal_embedding_name = 'normal_embedding.npz'

    normal_embedding_path = '{}/normal/{}'.format(embedding_path,normal_embedding_name)
    normal_data = np.load(normal_embedding_path)
    normal_entity_emb = normal_data['e_emb']
    

    for a in case_name:
        output_path =config['output_data_path_v'] 
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        output_path = os.path.join(output_path,a) # case name
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        label = []

        data_list = copy.deepcopy(normal_entity_emb)
        label.extend([0 for i in range(normal_entity_emb.shape[0])])

        
        anomalies_case_path = os.path.join(config['input_data_path'],a)
        # sort by start time (timestamp in case names)
        anomalies_case_list = os.listdir(anomalies_case_path)
        anomalies_case_list = sort_by_timestamp(anomalies_case_list)
        
        anomalies_output_path = '{}/{}'.format(embedding_path,a)
        for i in range(len(anomalies_case_list)):
            anomaly_embedding_path = os.path.join(anomalies_output_path,'embedding_anomaly_{}.npz'.format(anomalies_case_list[i]))

            data_2 = np.load(anomaly_embedding_path)
            anomaly_entity_emb = data_2['e_emb']
            data_list = np.concatenate((data_list,anomaly_entity_emb))
            label.extend([1 for i in range(anomaly_entity_emb.shape[0])])
            
        data_array = np.array(data_list)
        label = np.array(label)
        np.savez(os.path.join(output_path,'{}'.format(a)), y =label , x=data_array)
    
    # save normal data:
    output_path = config['output_data_path_v'] 
    output_path = os.path.join(output_path,'normal') # case name
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    label = np.array([0 for i in range(normal_entity_emb.shape[0])])
    np.savez(os.path.join(output_path,'normal'),y=label,x=normal_entity_emb)

