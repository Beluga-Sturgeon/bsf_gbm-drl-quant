# gbm.py

import torch
import numpy as np
from torch import tensor
from data import mean, stdev, standardize, normal, cumsum

OBS = 60
EXT = 20
EPOCH = 1000

def returns(raw):
    raw = tensor(raw)
    return (raw[1:] - raw[:-1]) / raw[:-1]

def vscore(raw, v, seed):
    raw = tensor(raw)
    for t in range(OBS - 1, len(raw)):
        temp = raw[t + 1 - OBS: t + 1]
        ret = returns(temp)
        s0 = temp[-1].item()
        mu = mean(ret.tolist())
        sigma = stdev(ret.tolist())
        drift = mu + 0.5 * sigma ** 2

        path = torch.zeros(EPOCH, EXT)
        normal(path, seed)
        cumsum(path)

        sum_val = 0
        for i in range(EPOCH):
            for j in range(EXT):
                path[i][j] *= sigma
                path[i][j] += drift * (j + 1)
                path[i][j] = s0 * torch.exp(path[i][j])
                sum_val += (path[i][j].item() > s0)

        v.append(sum_val / (EPOCH * EXT))

    standardize(v)
