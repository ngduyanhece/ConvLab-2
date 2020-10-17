import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
from torch import multiprocessing as mp
from convlab2.policy.ppo import PPO
from argparse import ArgumentParser

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    mp = mp.get_context('spawn')
except RuntimeError:
    pass

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batchsz", type=int, default=32, help="batch size of trajactory sampling")
    parser.add_argument("--epoch", type=int, default=200, help="number of epochs to train")
    parser.add_argument("--process_num", type=int, default=8, help="number of processes of trajactory sampling")
    args = parser.parse_args()

    policy_sys = PPO(True,dataset='rossatde')
    # s = {
    #     'utter': 'この前 の やつ でしょ 。'
    # }
    # s_vec = torch.Tensor(policy_sys.vector.state_vectorize(s))
    # a = policy_sys.predict(s)
    # print(a)
    