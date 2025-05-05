from Simulator import Simulator
from Scheduler import Scheduler, QueueKernelScheduler, KSMFScheduler, QueueScheduler, SequentialScheduler, QueueBasedScheduler, LumpScheduler
from Workload import SmallBankWorkload
from KernelWrapper import KernelWrapper
from Kernel import IntegerOptimisationKernelMkII
from utils import clone_transaction, conflict
import pickle
import numpy as np

workload = SmallBankWorkload()
probabilities = [0.15, 0.15, 0.15, 0.25, 0.15, 0.15] # https://github.com/cmu-db/benchbase/blob/main/config/mysql/sample_smallbank_config.xml#L22

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
workload_arr = []
T = 10
n_queues = 10
max_n_kernel = 10
filename = "arena_final_again_fixed_conflict.txt"
batch_size = 1
for t in range(T):
    num_txns = 1000
    random_transactions = workload.generate_transactions(num_txns, probabilities, start=t*num_txns)
    workload_arr.append([clone_transaction(txn) for txn in random_transactions])

class Contestant():
    def __init__(self, name:str, scheduler:Scheduler, model=None):
        self.name = name
        self.scheduler = scheduler
        self.sim = Simulator(scheduler, [])
        self.sim.model = model

contestants:list[Contestant] = []

# Load the trained models from pickle files
with open('trained_model_CNN.pkl', 'rb') as f:
    model_CNN = pickle.load(f)

with open('trained_model_linear.pkl', 'rb') as f:
    model_linear = pickle.load(f)

contestants.extend(
    [
        #Contestant("int-opt-k=10", QueueKernelScheduler(n_queues=n_queues, kernel=KernelWrapper(IntegerOptimisationKernelMkII(max_n_kernel)))),
        Contestant("k-smf", scheduler = KSMFScheduler(k=100)),
        #Contestant("queue-k-smf", scheduler = QueueScheduler(n_queues, KSMFScheduler, k=10)),
        #Contestant("queue-based", scheduler = QueueBasedScheduler(n_queues=n_queues)),
        Contestant("CNN Model", scheduler=LumpScheduler(), model=model_CNN),
        Contestant("Linear Model", scheduler=LumpScheduler(), model=model_linear),
    ]
)

done = False

for t in range(T):
    curdone = True
    for contestant in contestants:
        contestant.sim.add_transactions(workload_arr[t])
        if contestant.sim.model is not None:
            contestant.sim.sim(retryOnAbort=True, n=50, T=6)
        else:
            contestant.sim.sim(retryOnAbort=True)
        #print(contestant.sim.online_stats())
        d = contestant.sim.print_statistics()
        d["name"] = contestant.name
        curdone = curdone and contestant.sim.done()
        with open(filename, "a") as f:
            f.write(str(d) + "\n")
    done = done or curdone

# clean up the remaining
while not done:
    curdone = True
    for contestant in contestants:
        if contestant.sim.done(): continue
        if contestant.sim.model is not None:
            contestant.sim.sim(retryOnAbort=True, n=50, T=6)
        else:
            contestant.sim.sim(retryOnAbort=True)
        #print(contestant.sim.online_stats())
        d = contestant.sim.print_statistics()
        d["name"] = contestant.name
        curdone = curdone and contestant.sim.done()
        with open(filename, "a") as f:
            f.write(str(d) + "\n")
    done = done or curdone

# {'n_aborts': 0, 'n_successes': 10000, 'steps': 6369}
# {'n_aborts': 0, 'n_successes': 10000, 'steps': 6280}
