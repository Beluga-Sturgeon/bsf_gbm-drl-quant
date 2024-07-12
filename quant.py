import torch
import numpy as np
import subprocess
import os
import random
from threading import Thread
from net import Net, relu, relu_prime
from gbm import vscore, OBS, EPOCH, EXT
from data import read_csv, standardize, fix_dsp
import torch.nn.functional as F
import pandas as pd 
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

        self.init()

    def init(self, shape = [[500,500], [500,500], [500,500], [500,500], [500,500],[500,3]]):
        self.agent = Net()
        self.target = Net()
        for l in range(len(shape)):
            in_features, out_features = shape[l]
            self.agent.add_layer(in_features, out_features)
            self.target.add_layer(in_features, out_features)
        self.agent.init(seed=self.seed)
        self.sync()

    def generate_environment(self, ticker):
        cmd = f"./python/download.py {ticker} " + " ".join(self.indicators)
        print("running ", cmd)
        subprocess.run(cmd, shell=True)

        merge = "./data/merge.csv"
        raw = read_csv(merge)
        env = [[] for _ in range(len(raw) + 1)]

        def vscore_wrapper(raw_row, env_row, seed):
            vscore(raw_row, env_row, seed)
            print("finished vscore")


        for i in range(len(raw)):
            vscore_wrapper(raw[i], env[i + 1], self.seed)

        env[TICKER] = raw[TICKER][OBS - 1:]
        print("final env:", env)
        return env

    def sample_state(self, env, t):
        state = []
        for i in range(1, len(env)):
            dat = env[i][t + 1 - LOOK_BACK:t + 1]
            state.extend(dat)
        
        state_tensor = torch.tensor(state, dtype=torch.float32)  # Convert to a PyTorch tensor
        return state_tensor

    def greedy(self, state):
        
        q_values = self.agent.forward(state)
        q_values = q_values.detach().numpy()
        action = np.argmax(q_values)
        return action, q_values[action]

    def epsilon_greedy(self, state, eps):
        if random.random() < eps:
            print("(E) ", end='')
            action = random.randint(0, len(self.action_space) - 1)
            q_values = self.agent.forward(state)
            return action, q_values[action]
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

        print("Constants initialized")
        random.shuffle(self.tickers)
        for ticker in self.tickers:
            with open(f"./res/{ticker}_log", 'w') as out:
                header = "X," + ",".join(self.indicators) + ",action,benchmark,model\n"
                print(f"Writing header to file: {header}")
                
                # Write the header to the file
                out.write(header)
                out.flush()  # Ensure the buffer is flushed
                os.fsync(out.fileno()) 



                print(f"Processing ticker: {ticker}")
                benchmark = 1.0
                model = 1.0

                print("generating environment..")
                env = self.generate_environment(ticker)
                START = LOOK_BACK - 1
                END = len(env[TICKER]) - 2
                for t in range(START, END + 1):
                    if experiences <= CAPACITY:
                        EPS = (EPS_MIN - EPS_INIT) * experiences / CAPACITY + EPS_INIT
                        
                    state = self.sample_state(env, t)
                    action, q_value = self.epsilon_greedy(state, EPS)

                    next_state = self.sample_state(env, t + 1)
                    next_q_values = self.target.forward(next_state)

                    diff = (env[TICKER][t + 1] - env[TICKER][t]) / env[TICKER][t]
                    observed_reward = self.action_space[action] if diff >= 0.0 else -self.action_space[action]
                    optimal = observed_reward + GAMMA * max(next_q_values)

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

                    if experiences > CAPACITY:
                        break

                self.sync()
                out.close()
                subprocess.run(f"./python/log.py {ticker}-train", shell=True)

        self.save()
        print("done")

    def sgd(self, memory, alpha, LAMBDA):
        state = torch.tensor(memory.state, dtype=torch.float32)
        optimal = torch.tensor(memory.optimal, dtype=torch.float32)
        
        self.optimizer.zero_grad()
        q_values = self.agent(state)
        
        # Calculate the MSE loss
        loss = F.mse_loss(q_values[memory.action], optimal)

        # Regularization term for L2 regularization (weight decay)
        l2_reg = torch.tensor(0.0, dtype=torch.float32)
        for param in self.agent.parameters():
            l2_reg += torch.norm(param, 2)
        
        # Total loss includes MSE loss and L2 regularization term
        total_loss = loss + LAMBDA * l2_reg
        
        total_loss.backward()
        
        # Manually adjust learning rate by scaling gradients with alpha
        for param in self.agent.parameters():
            param.grad.data *= alpha
        
        self.optimizer.step()

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
                subprocess.run(f"./python/log.py {ticker}-test", shell=True)
                subprocess.run(f"./python/stats.py push {ticker}", shell=True)
                subprocess.run(f"./python/stats.py summary {ticker}", shell=True)
                subprocess.run(f"./python/analytics.py {ticker}", shell=True)

    def run(self):
        for ticker in self.tickers:
            env = self.generate_environment(ticker)
            state = self.sample_state(env, len(env[TICKER]) - 1)
            action = self.greedy(state)
            print(f"{ticker}: {action}")

    def sync(self):
        self.target.load_state_dict(self.agent.state_dict())

    def save(self):
        torch.save(self.agent.state_dict(), self.checkpoint)
        print(f"Model saved to {self.checkpoint}")

    def load(self):
        self.agent.load_state_dict(torch.load(self.checkpoint))
        self.sync()  # Synchronize the target network with the agent network
        print(f"Model loaded from {self.checkpoint}")

# Example usage:
if __name__ == "__main__":
    tickers = ["AAPL", "GOOGL", "MSFT"]
    indicators = ["indicator1", "indicator2"]
    seed = torch.manual_seed(42)  # Your seed value
    path = "./checkpoints/model_weights.txt"  # Your checkpoint file path
    shape = [(len(indicators) * LOOK_BACK, len(tickers))]  # Example shape, adjust as needed

    quant = Quant(tickers, indicators, seed, path)
    quant.init(shape)
    quant.build()
    quant.test()
    quant.run()
