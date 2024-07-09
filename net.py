# net.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def relu(x):
    return F.relu(x)

def relu_prime(x):
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

class Node:
    def __init__(self, in_features=None):
        self.b = 0.0
        self.s = 0.0
        self.z = 0.0
        self.e = 0.0
        self.w = torch.zeros(in_features) if in_features is not None else None

    def bias(self):
        return self.b

    def sum(self):
        return self.s

    def act(self):
        return self.z

    def err(self):
        return self.e

    def weight(self, index):
        return self.w[index]

    def init(self):
        self.s = self.z = self.e = 0.0

    def set_bias(self, val):
        self.b = val

    def set_sum(self, val):
        self.s = val

    def set_act(self, val):
        self.z = val

    def add_err(self, val):
        self.e += val

    def set_weight(self, index, val):
        self.w[index] = val

class Layer:
    def __init__(self, in_features=None, out_features=None):
        if in_features is not None and out_features is not None:
            self.in_features = in_features
            self.out_features = out_features
            self.nodes = [Node(in_features) for _ in range(out_features)]
        else:
            self.in_features = self.out_features = 0
            self.nodes = []

    def in_features(self):
        return self.in_features

    def out_features(self):
        return self.out_features

    def node(self, index):
        return self.nodes[index]

class Net:
    def __init__(self):
        self.layers = []

    def add_layer(self, in_features, out_features):
        self.layers.append(Layer(in_features, out_features))

    def init(self, seed):
        torch.manual_seed(seed)
        for layer in self.layers:
            for node in layer.nodes:
                std = np.sqrt(2.0 / layer.in_features)
                node.w = torch.randn(layer.in_features) * std

    def num_of_layers(self):
        return len(self.layers)

    def layer(self, index):
        return self.layers[index]

    def back(self):
        return self.layers[-1]

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float)
        for i, layer in enumerate(self.layers):
            y = torch.zeros(layer.out_features)
            for j, node in enumerate(layer.nodes):
                if i == 0:
                    dot = torch.dot(x, node.w)
                else:
                    dot = torch.dot(self.layers[i-1].nodes[j].act(), node.w)
                node.set_sum(dot + node.bias())
                if i == len(self.layers) - 1:
                    y[j] = node.sum()
                else:
                    node.set_act(relu(node.sum()))
            x = y
        return x.tolist()
