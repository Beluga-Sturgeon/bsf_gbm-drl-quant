import numpy as np
import pandas as pd
import torch

def read_csv(path):
    df = pd.read_csv(path)
    return df.values.T  # Transpose to match the original structure

def mean(data):
    return np.mean(data)

def stdev(data):
    return np.std(data)

def standardize(data):
    mu = mean(data)
    sigma = stdev(data)

    for i in range(len(data)):
        data[i] = (data[i] - mu) / sigma

def normal(mat, seed):
    torch.manual_seed(seed)
    normal_dist = torch.distributions.Normal(0, 1)
    mat_tensor = torch.as_tensor(mat, dtype=torch.float32)
    return normal_dist.sample(mat_tensor.shape).tolist()

def cumsum(mat):
    mat_tensor = torch.as_tensor(mat, dtype=torch.float32)
    cumsum_tensor = torch.cumsum(mat_tensor.clone().detach(), dim=1)  # Cumulative sum along the rows
    return cumsum_tensor.tolist()

def fix_dsp(path):
    import os
    if os.name == 'nt':  # For Windows
        return path.replace('/', '\\')
    return path
