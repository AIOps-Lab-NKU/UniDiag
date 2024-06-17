import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import os


class MyModel(nn.Module):
    def __init__(self, input_dim, output_dim, node_dim, final_dropout, saveDataPath, save=False):
        super(MyModel, self).__init__()
        self.input_dim = input_dim
        self.final_dropout = final_dropout
        self.save = save
        self.node_dim = node_dim
        self.output_dim = output_dim
        self.saveDataPath = saveDataPath

        # for attentional second-order pooling

        self.attend = nn.Linear(self.node_dim, int(input_dim / self.node_dim))
        self.linear1 = nn.Linear(self.input_dim, output_dim)

    def initialize(self):
        nn.init.normal_(self.attend.weight.data)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data)  # normal: mean=0, std=1

    def forward(self, data):
        graphs = torch.cat([data.x for data in data], dim=0)
        y = []
        if self.save:
            y = [int(d.y) for d in data]
    

        batch_graphs = torch.zeros(len(data), self.input_dim)
        batch_graphs = Variable(batch_graphs)

        for g_i in range(len(graphs)):
            np.set_printoptions(threshold=np.inf)
            cur_node_embeddings = torch.matmul(graphs[g_i].t(), graphs[g_i])  
            cur_node_embeddings = torch.round(cur_node_embeddings * 100) / 100
            attn_coef = self.attend(cur_node_embeddings)
            attn_weights = torch.transpose(attn_coef, 0, 1)
            cur_graph_embeddings = torch.matmul(attn_weights, cur_node_embeddings)
            batch_graphs[g_i] = cur_graph_embeddings.view(self.input_dim)

        if self.save:
            np.savez(self.saveDataPath, x=batch_graphs.detach().numpy(), y=y)

        score = F.dropout(self.linear1(batch_graphs), self.final_dropout, training=self.training)
        return score
