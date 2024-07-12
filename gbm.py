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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    raw = torch.as_tensor(raw, dtype=torch.float32).clone().detach().to(device)

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

        j_indices = torch.arange(EXT).float() + 1
        # Apply the transformations


        path = path * sigma
        path = path + drift * j_indices
        path = s0 * torch.exp(path)

        # Calculate sum_val
        sum_val = torch.sum(path > s0).item()

        v.append(sum_val / (EPOCH * EXT))
    standardize(v)
