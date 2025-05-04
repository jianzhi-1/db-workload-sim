import numpy as np
import matplotlib.pyplot as plt
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import itertools

from Workload import SmallBankWorkload
from utils import conflict
from models.CNNModel import CNNModel
from Simulator import Simulator
from Scheduler import LumpScheduler

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

# Hyperparameters to tune
hyperparams = {
    'batch_size': [5, 10, 20],
    'num_epochs': [30, 50, 100],
    'learning_rate': [0.001, 0.01, 0.1],
    'inner_channels': [[10, 5, 2], [16, 8, 4], [32, 16, 8]],
    'num_layers': [3, 4, 5]
}

# Fixed parameters
N = 50  # number of transactions we are considering in parallel
T = 6   # number of delays we care about [0, ..., T]
EPS = 1e-6

# Generate all combinations of hyperparameters
param_combinations = [dict(zip(hyperparams.keys(), v)) for v in itertools.product(*hyperparams.values())]

best_loss = float('inf')
best_params = None
best_model = None

for params in param_combinations:
    print(f"\nTraining with parameters: {params}")
    
    data_gen = generator(N, T)
    model = CNNModel(N=N, T=T, inner_channels=params['inner_channels'], num_layers=params['num_layers'])
    criterion = nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    loss_curve = []
    p_loss_curve = []
    lamb_loss_curve = []
    total_loss = 0.0

    for epoch in tqdm(range(params['num_epochs']), desc="training", unit="epoch"):
        x, random_transactions_list = data_gen(params['batch_size'])
        x = torch.from_numpy(x.astype(np.float32))

        optimizer.zero_grad()
        
        # Process each sample in the batch individually since CNN model expects single samples
        batch_lamb = []
        batch_p = []
        for b in range(params['batch_size']):
            sample = x[b]  # [N, N, 2*T + 1]
            lamb, p = model(sample)
            batch_lamb.append(lamb)
            batch_p.append(p)
        
        lamb = torch.stack(batch_lamb)  # [batch_size, N, T+2]
        p = torch.stack(batch_p)        # [batch_size, N, T+1]

        p_loss = 0.0
        lamb_loss = 0.0

        # Set up simulators and schedulers
        sim_list = []
        for i in range(params['batch_size']):
            scheduler = LumpScheduler()
            memory = dict()
            for j, txn in enumerate(random_transactions_list[i]):
                prob_arr = lamb[i][j].cpu().detach().numpy()
                memory[txn.txn] = np.random.choice(range(0, T+2), p=prob_arr, size=1)[0]
            scheduler.inject_memory(memory, T+1)
            sim_list.append(Simulator(scheduler, random_transactions_list[i]))

        # Run simulators to completion
        for i, sim in enumerate(sim_list):
            while not sim.done():
                sim.sim()

        # Calculate losses
        for b in range(params['batch_size']):
            for i, txn in enumerate(random_transactions_list[b]):
                sim = sim_list[b]
                t = sim.scheduler.memory[txn.txn]
                commit = (txn.txn in sim.result_statistics and sim.result_statistics[txn.txn] == 1)
                
                if t < T + 1:
                    if commit:
                        p_loss += -torch.log(F.sigmoid(p[b, i, t]))
                        target = lamb[b, i].clone().detach()
                        for tt in range(t + 1, T + 2):
                            target[tt] = EPS
                        target = target/torch.sum(target)
                        lamb_loss += F.kl_div(torch.log(lamb[b, i]), target)
                    else:
                        p_loss += -torch.log(1.0 - F.sigmoid(p[b, i, t]))
                        target = lamb[b, i].clone().detach()
                        target[t] = EPS
                        target = target/torch.sum(target)
                        lamb_loss += F.kl_div(torch.log(lamb[b, i]), target)

        loss = p_loss + lamb_loss
        loss.backward()
        optimizer.step()

        loss_curve.append(loss.item())
        p_loss_curve.append(p_loss.item())
        lamb_loss_curve.append(lamb_loss.item())
        total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}; Loss = {loss.item():.4f}; Accumulated Loss: {total_loss/(epoch + 1):.4f}")

    # Calculate average loss for this parameter combination
    avg_loss = total_loss / params['num_epochs']
    print(f"Average loss for these parameters: {avg_loss:.4f}")
    
    # Update best parameters if this combination is better
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_params = params
        best_model = model.state_dict()
        
        # Save the best model
        torch.save({
            'model_state_dict': model.state_dict(),
            'params': params,
            'loss': best_loss
        }, 'best_cnn_model.pth')

    # Plot loss curves for this parameter combination
    plt.figure(figsize=(15,10))
    plt.plot(np.arange(len(loss_curve)), np.array(loss_curve), label="loss")
    plt.plot(np.arange(len(p_loss_curve)), np.array(p_loss_curve), label="p_loss")
    plt.plot(np.arange(len(lamb_loss_curve)), np.array(lamb_loss_curve), label="lamb_loss")
    plt.xlabel('Epoch') 
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Training Loss Curve (params: {params})')
    plt.grid()
    plt.savefig(f'loss_curve_{hash(str(params))}.png')
    plt.close()

print(f"\nBest parameters found: {best_params}")
print(f"Best loss achieved: {best_loss:.4f}")
