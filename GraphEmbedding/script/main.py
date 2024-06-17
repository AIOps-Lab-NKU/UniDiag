import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from numpy import load
from torch import optim
import numpy as np

from GraphEmbedding.script.load_data import MyDataSet, load_data, separate_data, load_save_data, load_data_50
from GraphEmbedding.models.model import MyModel
from tqdm import tqdm


def train(model, train_graphs, optimizer, epoch):
    model.train()

    total_iters = 100
    loss_accum = 0
    for pos in range(total_iters):
        selected_idx = np.random.permutation(len(train_graphs))[:32]  # select random minibatch

        batch_graph = [train_graphs[idx] for idx in selected_idx]

        output = model(batch_graph)

        labels = torch.LongTensor([graph.y for graph in batch_graph])

        # compute loss
        l = nn.CrossEntropyLoss()
        loss = l(output, labels)

        # backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.detach().numpy()
        loss_accum += loss


    average_loss = loss_accum / total_iters

    return average_loss


def pass_data_iteratively(model, graphs, minibatch_size=64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i + minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)


def test(model, train_graphs, test_graphs):
    model.eval()
    output = pass_data_iteratively(model, train_graphs)

    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.y for graph in train_graphs])
    correct = pred.eq(labels.view_as(pred)).sum()
    acc_train = correct / float(len(train_graphs))

    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.y for graph in test_graphs])
    correct = pred.eq(labels.view_as(pred)).sum()
    acc_test = correct / float(len(test_graphs))
    return acc_train, acc_test


def save_data(model, data):
    graph = load_save_data(data)
    model.eval()
    model.save = True
    output = model(graph)


def graph_emb_main(config):
    config = config["graph_embedding"]
    trainDataPath = config["input_train_data_path"]
    allDataPath = config["input_all_data_path"]
    saveDataPath = config["output_path"]

    data = load(trainDataPath) 
 
    data = load_data(data)

    epochs = config["epoch_num"]
   
    output_vector_dimension = config["output_vector_dimension"]
    failure_classification_number = config["failure_classification_number"]
    node_dimension = config["node_dimension"]
     # Model parameters (output vector dimension, number of categories, node vector dimension, dropout of the last layer)
    model = MyModel(output_vector_dimension, failure_classification_number, node_dimension, 0.5, saveDataPath)
    model.initialize()

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    train_graphs, test_graphs = separate_data(data, 0, 5)
    max_acc = 0.0
    for epoch in tqdm(range(1, epochs + 1)):
        scheduler.step()
        avg_loss = train(model, data, optimizer, epoch)
        acc_train, acc_test = test(model, train_graphs, test_graphs)

        max_acc = max(max_acc, acc_test)
    s_data = load(allDataPath)
    save_data(model, s_data)
