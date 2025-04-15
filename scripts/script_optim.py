import numpy as np
import matplotlib.pyplot as plt

import logging

from Workload import SmallBankWorkload
from utils import conflict
from models import LinearModel
from Simulator import Simulator
from Scheduler import LumpScheduler
from Kernel import IntegerOptimisationKernel, SMFKernel

def generator(num_txns, T):
    workload = SmallBankWorkload()
    probabilities = [0.15, 0.15, 0.15, 0.25, 0.15, 0.15] # https://github.com/cmu-db/benchbase/blob/main/config/mysql/sample_smallbank_config.xml#L22
    def gen(batch_size):
        arr = np.zeros((batch_size, num_txns, num_txns, 2*T + 1))
        random_transactions_list = []
        for b in range(batch_size):
            random_transactions = workload.generate_random_transactions(num_txns, probabilities)
            random_transactions_list.append(random_transactions)
            for t in range(-T, T + 1, 1):
                for i, t1 in enumerate(random_transactions):
                    for j, t2 in enumerate(random_transactions):
                        arr[b][i][j][t + T] = conflict(t1, t2, t)
        return arr, random_transactions_list
    return gen

batch_size = 1 # batch size
N = 50 # number of transactions we are considering in parallel
T = 6 # number of delays we care about [0, ..., T]
num_epochs = 50
EPS = 1e-6

data_gen = generator(N, T)
C, random_transactions_list = data_gen(batch_size)

kernel = IntegerOptimisationKernel(N, T)
throughput, res = kernel.run(C, debug=True)
print(f"Number of transactions scheduled = {throughput}")
print(res)

# smf kernel for the same workload
smf_kernel = SMFKernel(N, T)
throughput, res = smf_kernel.run(C, debug=True)
print(f"Number of transactions scheduled = {throughput}")
print(res)
