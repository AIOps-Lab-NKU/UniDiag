# entity embedding full process
import argparse
import os
import shutil
import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[0]))
from data_process import for_GAIA_process
from case_main import for_GAIA_emb
from case_vector import for_GAIA_vector
from for_graph import for_GAIA_graph

def entity_emb_main(config):
    dataset_choice = config['dataset']
    entity_emb_config = config['entity_embedding']
    # copy data:
    source = config['data_path']
    target = entity_emb_config['input_data_path']
    
    if os.path.exists(target):
        shutil.rmtree(target)
    try:
        shutil.copytree(source,target)
    except:
        print("Unable to copy file.",sys.exc_info())
        sys.exit()
    
    # data process:
    if dataset_choice == 'gaia':
        for_GAIA_process(target,entity_emb_config['log_path']+'/'+'gaia_process.txt')
        print("GAIA data process √")
    
    
    # RE-GCN
    # train:
    if not os.path.exists(entity_emb_config['output_data_path']):
        os.makedirs(entity_emb_config['output_data_path'])
    train_output_embedding = os.path.join(entity_emb_config['output_data_path'],'normal')
    if not os.path.exists(train_output_embedding):
        os.makedirs(train_output_embedding)
    command = "python3 {} --path_normal {}  -d normal  --model_name {} --model_path {} --nodes_list {} --n-hidden {} --n-layers {} --n-epochs {} --layer-norm --weight 0.5 --entity-prediction --relation-prediction --output_embedding {} --train-history-len {} --test-history-len {} > {} ".format(entity_emb_config['RE-GCN_main'],entity_emb_config['input_data_path'],entity_emb_config['model_name'],entity_emb_config['model_path'],entity_emb_config['nodes'],entity_emb_config['hidden_dimension'],entity_emb_config['gcn_layer'],entity_emb_config['epoch'],train_output_embedding,entity_emb_config['window_size'],entity_emb_config['window_size'],entity_emb_config['log_path']+'/'+dataset_choice+'_train.txt')
    res = os.system(command)
    print("Entity Embedding Model Train √")
    
    # case embedding:
    # confirm model:
    if not os.path.exists(entity_emb_config['model_path']+entity_emb_config['model_name']):
        raise Exception("Unable to embed!")
    if dataset_choice == 'gaia':
        # case embedding:
        for_GAIA_emb(entity_emb_config)
        print("GAIA cases embedding √ ")
        # vector combination:
        for_GAIA_vector(entity_emb_config)
        print("GAIA vector combination √ ")
        # for graph train and emb:
        for_GAIA_graph(entity_emb_config)
        print("GAIA vector to graph input files √ ")


    
        


    




    



    
       

