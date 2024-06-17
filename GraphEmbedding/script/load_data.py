import random

import numpy as np
import torch
from numpy import load, size
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn import preprocessing

class graph:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class MyDataSet(Dataset):
    def __init__(self, loaded_data):
        self.x_data = loaded_data['x']
        self.y_data = loaded_data['y']

        self.length = len(self.y_data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]


def separate_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = KFold(n_splits=10, shuffle=True, random_state=seed)

    Y = [np.array(graph.y) for graph in graph_list]
    X = [graph.x for graph in graph_list]

    l = np.zeros(shape=(len(X), 1))

    idx_list = []
    idy_list = []

    for idx, idy in skf.split(np.array(Y)):
        idx_list.append(idx)
        idy_list.append(idy)

    train_idx = idx_list[fold_idx]
    test_idy = idy_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idy]

    return train_graph_list, test_graph_list


def load_data(data):
    my_dataset = MyDataSet(data)
    train_loader = DataLoader(dataset=my_dataset,  # the dataset to pass
                              batch_size=1,  # the size of a mini-batch
                              shuffle=True,  # Whether the order of the data sets should be disturbed is generally required.
                              num_workers=0)

    graphs = []
    for i, (x, y) in enumerate(train_loader):
        y = y.to(torch.int)
        x = x.to(torch.float32)
        graphs.append(graph(x, y))

    return graphs


def load_data_50(data):
    index_0 = list(data['y']).index(0)
    index_1 = list(data['y']).index(1)
    index_2 = list(data['y']).index(2)
    index_3 = list(data['y']).index(3)
    index_4 = list(data['y']).index(4)
    my_dataset = MyDataSet(data)
    train_loader = DataLoader(dataset=my_dataset,
                              batch_size=1, 
                              shuffle=False,  
                              num_workers=0)
    train_graphs = []
    test_graphs = []
    for i, (x, y) in enumerate(train_loader):
        if i in range(index_0, index_0 + 50) or i in range(index_1, index_1 + 50) or i in range(index_2,
                                                                                                index_2 + 50) or i in range(
                index_3, index_3 + 50) or i in range(index_4, index_4 + 50):
            train_graphs.append(graph(x, y))
        else:
            test_graphs.append(graph(x, y))



    return train_graphs, test_graphs


def load_save_data(data):
    my_dataset = MyDataSet(data)
    train_loader = DataLoader(dataset=my_dataset, 
                              batch_size=1,  
                              shuffle=False,
                              num_workers=0)

    graphs = []
    for i, (x, y) in enumerate(train_loader):
        x = x.to(torch.float32)
        graphs.append(graph(x, y))
    return graphs
