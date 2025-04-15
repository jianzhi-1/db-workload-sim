import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

import logging

from Workload import SmallBankWorkload
from utils import conflict
from models import LinearModel
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

batch_size = 1 # batch size
N = 50 # number of transactions we are considering in parallel
T = 6 # number of delays we care about [0, ..., T]
num_epochs = 50
EPS = 1e-6

data_gen = generator(N, T)

x, random_transactions_list = data_gen(batch_size)
print(x)
print(x.shape)

"""
model = LinearModel(N, T).to("cuda:0") # PUT MODEL ON CUDA
criterion = nn.CrossEntropyLoss(reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_curve = []
p_loss_curve = []
lamb_loss_curve = []
total_loss = 0.0

for epoch in tqdm.notebook.trange(num_epochs, desc="training", unit="epoch"):

    x, random_transactions_list = data_gen(batch_size)
    assert x.shape == (batch_size, N, N, 2*T + 1)
    x = torch.from_numpy(x.astype(np.float32)).to("cuda:0") # PUT X ON CUDA

    optimizer.zero_grad()
    lamb, p = model(x)

    assert lamb.shape == (batch_size, N, T+2)
    assert p.shape == (batch_size, N, T+1)
    p_loss = 0.0
    lamb_loss = 0.0

    # Suppose set up batch_size simulators and schedulers and got back a list of which transactions aborted...

    sim_list = []
    for i in range(batch_size):
        scheduler = LumpScheduler()
        memory = dict()
        for j, txn in enumerate(random_transactions_list[i]):
            prob_arr = lamb[i][j].cpu().detach().numpy()
            memory[txn.txn] = np.random.choice(range(0, T+2), p=prob_arr, size=1)[0]
        scheduler.inject_memory(memory, T+1)
        sim_list.append(Simulator(scheduler, random_transactions_list[i]))

    # run simulators to completion
    for i, sim in enumerate(sim_list):
        while not sim.done():
            sim.sim()
        if i == 0: sim.print_statistics()

    for b in range(batch_size):
        for i, txn in enumerate(random_transactions_list[b]):
            sim = sim_list[b]
            t = sim.scheduler.memory[txn.txn] # when the transaction is scheduled
            commit = (txn.txn in sim.result_statistics and sim.result_statistics[txn.txn] == 1) # whether the transaction is committed
            assert t >= 0 and t <= T+1, "t is outside the range of [0, T+1]"
            if t < T + 1: # i.e. actually scheduled
                if commit:
                    # batch b, transaction i, scheduled at time t committed
                    p_loss += -torch.log(F.sigmoid(p[b, i, t]))
                    target = lamb[b, i].clone().detach()
                    for tt in range(t + 1, T + 2):
                        target[tt] = EPS
                    target = target/torch.sum(target)
                    lamb_loss += F.kl_div(torch.log(lamb[b, i]), target)
                else: # aborted
                    # batch b, transaction i
                    p_loss += -torch.log(1.0 - F.sigmoid(p[b, i, t]))
                    target = lamb[b, i].clone().detach()
                    target[t] = EPS # reset the probability for that value
                    target = target/torch.sum(target)
                    lamb_loss += F.kl_div(torch.log(lamb[b, i]), target)
            else:
                pass # not scheduled

    loss = p_loss + lamb_loss

    loss.backward()
    optimizer.step()

    loss_curve.append(loss.item())
    p_loss_curve.append(p_loss.item())
    lamb_loss_curve.append(lamb_loss.item())

    total_loss += loss.item()

    if epoch % 1 == 0:
        print(f"Epoch {epoch}; Loss = {loss.item():.4f}; Accumulated Loss: {total_loss/(epoch + 1):.4f}")

# plt.figure(figsize=(15,10))
# plt.plot(np.arange(len(loss_curve)), np.array(loss_curve), label="loss")
# plt.plot(np.arange(len(p_loss_curve)), np.array(p_loss_curve), label="p_loss")
# plt.plot(np.arange(len(lamb_loss_curve)), np.array(lamb_loss_curve), label="lamb_loss")
# plt.xlabel('Epoch') 
# plt.ylabel('Loss')
# plt.legend()
# plt.title('Training Loss Curve')
# plt.grid()
# plt.show()
"""