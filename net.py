import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def relu(x):
    return F.relu(x)

def relu_prime(x):
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()

    def add_layer(self, in_features, out_features):
        self.layers.append(nn.Linear(in_features, out_features))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = relu(layer(x))
        x = self.layers[-1](x)  # No activation on the output layer
        return x

    def init(self, seed):
        torch.manual_seed(seed)
        for layer in self.layers:
            layer:nn.Linear
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)



    def get_layer_output(self, x, layer_index):
        for i, layer in enumerate(self.layers):
            x = relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
            if i == layer_index:
                break
        return x
