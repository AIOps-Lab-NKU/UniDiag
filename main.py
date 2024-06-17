import argparse
import os
import yaml
from EntityEmbedding.script.entity_emb_main import entity_emb_main
from GraphEmbedding.script.main import graph_emb_main
from Cluster.cluster import cluster_main

def load_yaml(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    return config

parser = argparse.ArgumentParser()
parser.add_argument('--config',help='config.yml path')
parser.add_argument('--dataset',help='gaia or others')
args = parser.parse_args()

# load yaml
config_a = load_yaml(args.config)
if args.dataset == 'gaia':
    config = config_a['gaia']

# entity embedding
entity_emb_main(config)

# graph embedding:
graph_emb_main(config)

# cluster
cluster_main(config)
