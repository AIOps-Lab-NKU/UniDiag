import os
import time

"""
GAIA DIR structure
GAIA
    - normal
    - access_permission_denied
    - file_moving
    - memory_anomalies 
    - normal_memory_freed_label
"""

def for_GAIA_emb(config):
    case_name = config['failure_type']
    for i in case_name:
        output_path = config['output_data_path']
        embedding_output_path = os.path.join(output_path,i) 
        if not os.path.exists(embedding_output_path):
            os.mkdir(embedding_output_path)

        anomalies_path = os.path.join(config['input_data_path'],i)
        anomalies_case_list = sorted(os.listdir(anomalies_path))

        for j in range(len(anomalies_case_list)):
            if os.path.exists(os.path.join(embedding_output_path,'embedding_anomaly_{}.npz'.format(anomalies_case_list[j]))):
                continue
            command = "python3 {} --path_normal {}  -d normal --test --path_anomaly {} --dataset_anomaly '{}' --model_name {} --model_path {} --nodes_list {}  --n-hidden {} --n-layers {} --layer-norm --weight 0.5 --entity-prediction --relation-prediction --output_embedding {} --test-history-len {} >> {} ".format(config['RE-GCN_main'],config['input_data_path'],anomalies_path,anomalies_case_list[j],config['model_name'],config['model_path'],config['nodes'],config['hidden_dimension'],config['gcn_layer'],embedding_output_path,config['window_size'],config['log_path']+'/'+'gaia_emb.txt')
            res = os.system(command)
            print("{} finish======√".format(anomalies_case_list[j]))
            
