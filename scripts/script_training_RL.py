
import pickle
import random
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

from Workload import SmallBankWorkload
from models.QNet import Trainer
from utils import conflict

def generator(num_txns, T):
    corr_params = {
        "normal_scale": 10.0,
        "event_scale": 20.0,
        "correlation_factor": 0.1
    }
    workload = SmallBankWorkload(correlated=True, corr_params=corr_params)
    probabilities = [0.15, 0.15, 0.15, 0.25, 0.15, 0.15] # https://github.com/cmu-db/benchbase/blob/main/config/mysql/sample_smallbank_config.xml#L22
    def gen(batch_size, eps=0.1):
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

N = 50
T = 6
EPISODES = 100
GAMMA = 0.9 # discount factor
EPSILON = 0.1
LR = 1e-3
batch_size = 1
    
def print_conflict_matrix(conflict_tensor, transactions):
    """
    Print raw conflict matrix for each transaction.

    Args:
        conflict_tensor: torch tensor of shape (batch, N, N, 2T+1)
        transactions: list of Transaction objects
    """
    # Take first batch and convert to numpy
    conflict_matrix = conflict_tensor[0].cpu().numpy()
    N = conflict_matrix.shape[0]
    T = (conflict_matrix.shape[2] - 1) // 2
    print(conflict_matrix.shape)

    print("\n" + "="*80)
    print("CONFLICT MATRIX ANALYSIS")
    print("="*80)

    for i in range(N):
        print("\n" + "-"*80)
        print(f"Transaction {i}: {transactions[i]}")
        print("-"*80)

         # Print header for timesteps
        header = "j\\t | " + " | ".join([f"{t-T:3d}" for t in range(2*T+1)])
        print(header)
        print("-" * len(header))

        # Print conflict matrix for this transaction
        for j in range(N):
            row = f"{j} | " + " | ".join([f"{round(conflict_matrix[i,j,t])}" for t in range(2*T+1)])
            print(row)

data_gen = generator(N, T)
# Instantiate scheduler
trainer = Trainer(N=N, T=T, lr=LR, gamma=GAMMA, epsilon=EPSILON)
# Training loop
reward_history = []

for episode in tqdm(range(EPISODES), desc="training", unit="epoch"):
    conflict_matrix, random_transactions_list = data_gen(batch_size)
    conflict_matrix = torch.tensor(conflict_matrix)
    #print_conflict_matrix(conflict_matrix, random_transactions_list[0])
    total_reward = trainer.run_episode(conflict_matrix)
    reward_history.append(total_reward)

    if (episode + 1) % 50 == 0:
        avg_reward = sum(reward_history[-50:]) / 50
        print(f"Episode {episode + 1}: Avg Reward (last 50): {avg_reward:.2f}", flush=True)

# Plot illustrations
plt.figure(figsize=(15,10))
plt.plot(np.arange(len(reward_history)), np.array(reward_history), label="reward_history")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.title("Reward History Curve")
plt.grid()
plt.show()

with open("model_RL.pkl", "wb") as f:
    pickle.dump(trainer.q_net, f)
