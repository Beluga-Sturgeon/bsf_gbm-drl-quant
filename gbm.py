import torch
import numpy as np
from data import mean, stdev, normal, cumsum, standardize

OBS = 60
EXT = 20
EPOCH = 1000

def returns(raw):
    raw = torch.tensor(raw)
    return (raw[1:] - raw[:-1]) / raw[:-1]

def vscore(raw, v, seed):
    torch.manual_seed(seed)
    raw = torch.as_tensor(raw, dtype=torch.float32).clone().detach()

    for t in range(OBS - 1, len(raw)):
        temp = raw[t + 1 - OBS: t + 1]
        ret = returns(temp)
        s0 = temp[-1].item()
        mu = mean(ret.tolist())
        sigma = stdev(ret.tolist())
        drift = mu + 0.5 * sigma ** 2

        path = torch.zeros(EPOCH, EXT, dtype=torch.float32)
        path = normal(path, seed)
        path = cumsum(path)

        sum_val = 0
        for i in range(EPOCH):
            for j in range(EXT):
                path[i][j] *= sigma
                path[i][j] += drift * (j + 1)
                print(path[i][j])
                print(type(path[i][j]))
                path[i][j] = s0 * torch.exp(path[i][j])  # Ensure path[i][j] is a tensor
                sum_val += (path[i][j].item() > s0)

        v.append(sum_val / (EPOCH * EXT))

    print(f"v before standardize: {v}")
    standardize(v)
    print(f"v after standardize: {v}")
