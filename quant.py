import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import subprocess
import os
import random
from threading import Thread

LOOK_BACK = 100
TICKER = 0

class Memory:
    def __init__(self, state, action, optimal):
        self.state = state
        self.action = action
        self.optimal = optimal

class Quant:
    def __init__(self, tickers, indicators, seed, path):
        self.tickers = tickers
        self.indicators = indicators
        self.agent = None
        self.target = None
        self.checkpoint = path
        self.seed = seed
        self.action_space = [-1.0, 0.0, 1.0]  # short, idle, long

    def init(self, shape):
        self.agent = nn.Sequential()
        self.target = nn.Sequential()
        for l in range(len(shape)):
            in_features, out_features = shape[l]
            self.agent.add_module(f"fc{l}", nn.Linear(in_features, out_features))
            self.target.add_module(f"fc{l}", nn.Linear(in_features, out_features))
        self.sync()

    def sync(self):
        for target_param, param in zip(self.target.parameters(), self.agent.parameters()):
            target_param.data.copy_(param.data)

    def generate_environment(self, ticker):
        cmd = f"python3 ./python/download.py {ticker} " + " ".join(self.indicators)
        subprocess.run(cmd, shell=True)

        merge = "./data/merge.csv"
        raw = np.genfromtxt(merge, delimiter=',')
        env = [[] for _ in range(raw.shape[0] + 1)]

        threads = []
        for i in range(raw.shape[0]):
            th = Thread(target=self.vscore, args=(raw[i], env[i + 1]))
            threads.append(th)
            th.start()
        for th in threads:
            th.join()

        env[TICKER] = raw[TICKER][LOOK_BACK - 1:]

        return env

    def sample_state(self, env, t):
        state = []
        for i in range(1, len(env)):
            dat = env[i][t + 1 - LOOK_BACK:t + 1]
            state.extend(dat)
        return state

    def greedy(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.agent(state_tensor)
        _, action = torch.max(q_values, 1)
        return action.item()

    def epsilon_greedy(self, state, eps):
        if random.random() < eps:
            print("(E) ", end='')
            return random.randint(0, len(self.action_space) - 1)
        else:
            print("(P) ", end='')
            return self.greedy(state)

    def build(self):
        EPS_INIT = 1.0
        EPS_MIN = 0.1
        GAMMA = 0.8
        ALPHA = 0.000001
        LAMBDA = 0.1

        replay = []
        CAPACITY = 25000
        BATCH_SIZE = 10

        EPS = EPS_INIT
        RSS = 0.0
        MSE = 0.0
        experiences = 0

        random.shuffle(self.tickers)
        for ticker in self.tickers:
            with open(f"./res/{ticker}_log", 'w') as out:
                out.write("X," + ",".join(self.indicators) + ",action,benchmark,model\n")
                benchmark = 1.0
                model = 1.0

                env = self.generate_environment(ticker)
                START = LOOK_BACK - 1
                END = len(env[TICKER]) - 2
                for t in range(START, END + 1):
                    if experiences <= CAPACITY:
                        EPS = (EPS_MIN - EPS_INIT) * experiences / CAPACITY + EPS_INIT
                    state = self.sample_state(env, t)
                    action = self.epsilon_greedy(state, EPS)
                    q_value = self.agent(torch.FloatTensor(state).unsqueeze(0))[0, action].item()

                    next_state = self.sample_state(env, t + 1)
                    next_q_values = self.target(torch.FloatTensor(next_state).unsqueeze(0))

                    diff = (env[TICKER][t + 1] - env[TICKER][t]) / env[TICKER][t]
                    observed_reward = self.action_space[action] if diff >= 0.0 else -self.action_space[action]
                    optimal = observed_reward + GAMMA * torch.max(next_q_values).item()

                    benchmark *= 1.0 + diff
                    model *= 1.0 + diff * self.action_space[action]
                    out.write(",".join(map(str, state[LOOK_BACK - 1::LOOK_BACK])) + f",{action},{benchmark},{model}\n")

                    RSS += (optimal - q_value) ** 2
                    experiences += 1
                    MSE = RSS / experiences

                    print(f"LOSS={MSE} EPS={EPS} ALPHA={ALPHA} ", end='')
                    print(f"T={t} @ {ticker} ACTION={action} -> OBS={observed_reward} OPT={optimal} ", end='')
                    print(f"BENCH={benchmark} MODEL={model}")

                    replay.append(Memory(state, action, optimal))

                    if len(replay) == CAPACITY:
                        indices = random.sample(range(len(replay)), BATCH_SIZE)
                        for k in indices:
                            self.sgd(replay[k], ALPHA, LAMBDA)
                        replay = replay[BATCH_SIZE:]

                self.sync()
                out.close()
                subprocess.run(f"python3 ./python/log.py {ticker}-train", shell=True)

        self.save()

    def sgd(self, memory, alpha, LAMBDA):
        state_tensor = torch.FloatTensor(memory.state).unsqueeze(0)
        q_values = self.agent(state_tensor)
        loss = (memory.optimal - q_values[0, memory.action]) ** 2
        self.agent.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in self.agent.parameters():
                param -= alpha * param.grad
                param -= LAMBDA * param

    def test(self):
        for ticker in self.tickers:
            with open(f"./res/{ticker}_log", 'w') as out:
                out.write("X," + ",".join(self.indicators) + ",action,benchmark,model\n")
                benchmark = 1.0
                model = 1.0

                env = self.generate_environment(ticker)
                START = LOOK_BACK - 1
                END = len(env[TICKER]) - 2
                for t in range(START, END + 1):
                    state = self.sample_state(env, t)
                    action = self.greedy(state)

                    diff = (env[TICKER][t + 1] - env[TICKER][t]) / env[TICKER][t]
                    benchmark *= 1.0 + diff
                    model *= 1.0 + diff * self.action_space[action]

                    out.write(",".join(map(str, state[LOOK_BACK - 1::LOOK_BACK])) + f",{action},{benchmark},{model}\n")
                    print(f"T={t} @ {ticker} ACTION={action} DIFF={diff} BENCH={benchmark} MODEL={model}")

                out.close()
                subprocess.run(f"python3 ./python/log.py {ticker}-test", shell=True)
                subprocess.run(f"python3 ./python/stats.py push {ticker}", shell=True)
                subprocess.run(f"python3 ./python/stats.py summary {ticker}", shell=True)
                subprocess.run(f"python3 ./python/analytics.py {ticker}", shell=True)

    def run(self):
        for ticker in self.tickers:
            env = self.generate_environment(ticker)
            state = self.sample_state(env, len(env[TICKER]) - 1)
            action = self.greedy(state)
            print(f"{ticker}: {action}")

    def save(self):
        with open(self.checkpoint, 'w') as out:
            for param in self.agent.parameters():
                out.write(" ".join(map(str, param.data.numpy().flatten())) + "\n")

    def load(self):
        with open(self.checkpoint, 'r') as inp:
            for param, line in zip(self.agent.parameters(), inp):
                data = list(map(float, line.strip().split()))
                param.data.copy_(torch.tensor(data))

    def vscore(self, raw, env):
        # Dummy method for VS score calculation
        pass

# Example usage:
if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL"]
    indicators = ["indicator1", "indicator2"]
    seed = torch.manual_seed(42)
    path = "./checkpoint.pth"

    model = Quant(tickers, indicators, seed, path)
    model.init([[500, 500], [500, 500], [500, 500], [500, 500], [500, 500], [500, 3]])
    model.build()
    model.test()
    model.run()
