## Quick Start

### Environment & dependencies

```
conda create -n UniDiag python=3.7

conda activate UniDiag

pip install -r requirements.txt

conda install cudatoolkit=10.2(Optional)
```
### Run

- gaia:
```
cd UniDiag-main
python3 main.py --dataset gaia --config config.yml
```


## Folder Structure
```
UniDiag:
│  config.yml   
│  main.py                  # Main Function
│  README.md
│  requirements.txt 
│  
├─Cluster                   # Clustering
│
├─data                      # Pre-built TKG dataset.
│  └─GAIA_TKG
│                  
├─EntityEmbedding           # Entity Embedding 
│  ├─data                   # Processed TKG data for model input
│  ├─log            
│  ├─models                 # trained RE-GCN model
│  ├─output                 # intermediate files
│  ├─rgcn                   # R-GCN model
│  ├─script                 # scripts for entity embedding process
│  └─src                    # RE-GCN model
│        
├─GraphEmbedding            # Graph Embedding 
│  ├─models                 # model for Graph Embedding
│  └─script                 # model training
│          
├─output                    # Entity Embedding, Graph Embedding and Clustering results
│  ├─GAIA_cluster           # Clustering result
│  │      GAIA_result.csv   # Raw result
│  │      GAIA_stat.csv     # 10 snapshots to 1 result
│  │      report.txt        # NMI, ACC, Precision, Recall and F1-score
│  │      
│  └─GAIA_graph             # Embedding Files
│          GAIA_all.npz     # All Entity Embeddings
│          GAIA_save.npz    # Graph Embeddings
│          GAIA_train.npz   # Entity Embeddings for Graph model training
│          
└─Preprocessed      
        data_solver.py
        detection.py        
        recommend.py
        result.csv


```

## AIopsDataset

The  AIops Dataset is public at : [https://1drv.ms/u/s!An9w77NkXc7par4z4IUZIWOqHDA?e=UcFgj9](https://1drv.ms/u/s!An9w77NkXc7paraDgzcP_MutHIo?e=Zky3vh)

The AIops Dataset comes from a simulated e-commerce system based on a microservices architecture. The system is deployed on the CCB cloud, and its traffic is consistent with real-world business traffic. The fault scenarios are derived from fault types summarized from the real system, and are replayed in batches.

The system is based on Hipster Shop, an open-source project from Google composed of microservices written in various languages. It employs a dynamic deployment architecture, consisting of 10 core services and 6 virtual machines. Each service is deployed with 4 pods, for a total of 40 pods. These 40 pods are dynamically deployed on 6 virtual machines.

For the fault scenarios in the competition, faults are injected at three levels: service, pod, and node. Faults at the service and pod levels mainly target container faults in Kubernetes, including network latency, packet loss, resource packet damage, repeated resource packet transmission, sudden CPU and memory pressure, disk read pressure, disk write pressure, and process stoppage.

The dataset includes the dynamic topology of the application service, real-time trace data, real-time business metrics, performance metrics, and logs. Performance metrics come from containers, operating systems, and the JVM, among others.


