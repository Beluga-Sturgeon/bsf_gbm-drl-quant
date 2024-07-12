#!/usr/bin/env python3

import sys
import pandas as pd
import matplotlib.pyplot as plt
import os

def remove_suffix(string, suffix):
    if string.endswith(suffix):
        return string[:-len(suffix)]
    return string

def process_string(string):
    if string.endswith('-test'):
        return remove_suffix(string, '-test')
    elif string.endswith('-train'):
        return remove_suffix(string, '-train')
    return string



path = sys.argv[1]
ticker = process_string(path)
if not os.path.exists(f"./res/{ticker}_log"):
    with open(f"./res/{ticker}_log", 'w') as file:
        file.write('')

log = pd.read_csv(f"./res/{ticker}_log")

plt.figure(figsize=(30,20))

plot_index = [1, 2, 4, 6, 8]
colors = ["cadetblue", "steelblue", "dodgerblue", "lightskyblue", "cornflowerblue"]

k = 0
for key in log.columns[:5]:
    plt.subplot(4, 2, plot_index[k])
    plt.title(f"State: {key}")
    plt.plot(log[key], label=key, color=colors[k])
    plt.legend()
    k += 1

#####

plt.subplot(4, 2, 3)
plt.title("Action")
plt.plot(log["action"], label="0: Short\n1: Idle\n2: Long", color="steelblue")
plt.legend()

plt.subplot(4, 2, 5)
plt.title("Reward (Historical)")
plt.plot(log["benchmark"], label="Benchmark", color="cadetblue")
plt.plot(log["model"], label="Model", color="steelblue")
plt.legend()

#####

benchmark = log["benchmark"][-252:]
model = log["model"][-252:]

benchmark = (benchmark - benchmark.iloc[0]) * 100 / benchmark.iloc[0]
model = (model - model.iloc[0]) * 100 / model.iloc[0]

plt.subplot(4, 2, 7)
plt.title("Reward (Recent)")
plt.plot(benchmark, label="Benchmark", color="cadetblue")
plt.plot(model, label="Model", color="steelblue")
plt.legend()

plt.savefig("./res/{}.png" .format(path))
