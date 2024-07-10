import sys
import os
import random
import subprocess
import torch

from quant import Quant  # Assuming Quant class is defined in quant.py

tickers = []
indicators = ["SPY", "IEF", "EUR=X", "CL=F"]
mode = ""
checkpoint = "./checkpoint.pth"  # Default checkpoint path
seed = torch.manual_seed(random.randint(1, 10000))
seed = random.randint(1, 10000)

def boot(args):
    global mode, tickers, checkpoint
    device = 0 if torch.cuda.is_available() else "cpu"
    if device == 0:
        torch.cuda.set_device(0)
    print("Using deivce: ", device)
    print(args)
    mode = args[1]
    tickers = args[2:-1]
    checkpoint = args[-1]

    # Clear data and result directories
    subprocess.run(["rm", "-rf", "./data/*", "./res/*"])

    print(f"Mode: {mode}")
    print(f"Tickers: {tickers}")
    print(f"Checkpoint: {checkpoint}")

def main():
    boot(sys.argv)

    quant = Quant(tickers, indicators, seed, checkpoint)

    if mode == "build":
        quant.build()
    elif mode == "test":
        quant.test()
    elif mode == "run":
        quant.run()
    else:
        print("Invalid mode. Please use 'build', 'test', or 'run'.")

if __name__ == "__main__":
    main()
