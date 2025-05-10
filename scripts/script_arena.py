import pickle
from Simulator import Simulator
from Scheduler import LumpScheduler, RLSMFTwistedScheduler, Scheduler, QueueKernelScheduler, KSMFScheduler, QueueScheduler, SequentialScheduler, QueueBasedScheduler, RLSMFTwistedScheduler
from Workload import SmallBankWorkload
from KernelWrapper import KernelWrapper
from Kernel import IntegerOptimisationKernelMkII
from utils import clone_transaction

workload = SmallBankWorkload()
probabilities = [0.15, 0.15, 0.15, 0.25, 0.15, 0.15] # https://github.com/cmu-db/benchbase/blob/main/config/mysql/sample_smallbank_config.xml#L22

workload_arr = []
T = 20
n_queues = 10
max_n_kernel = 10
filename = "arena_final_again_fixed_conflict.txt"
for t in range(T):
    num_txns = 1000
    random_transactions = workload.generate_transactions(num_txns, probabilities, start=t*num_txns)
    workload_arr.append([clone_transaction(txn) for txn in random_transactions])

class Contestant():
    def __init__(self, name:str, scheduler:Scheduler, model=None):
        self.name = name
        self.scheduler = scheduler
        self.sim = Simulator(scheduler, [], False)
        self.sim.model = model

contestants:list[Contestant] = []

# Load the trained models from pickle files
#with open('trained_model_CNN.pkl', 'rb') as f:
#    model_CNN = pickle.load(f)

with open('trained_model_linear.pkl', 'rb') as f:
   model_linear = pickle.load(f)

with open('model_RL.pkl', 'rb') as f:
    model_RL = pickle.load(f)
    
contestants.extend(
    [
        #Contestant("int-opt-k=10", QueueKernelScheduler(n_queues=n_queues, kernel=KernelWrapper(IntegerOptimisationKernelMkII(max_n_kernel)))),
        #Contestant("k-smf", scheduler = KSMFScheduler(k=100)),
        #Contestant("queue-k-smf", scheduler = QueueScheduler(n_queues, KSMFScheduler, k=10)),
        #Contestant("queue-based", scheduler = QueueBasedScheduler(n_queues=n_queues))
        Contestant("Linear Model", scheduler=LumpScheduler(), model=model_linear),
        #Contestant("RL Model", scheduler=LumpScheduler(), model=model_RL),

    ]
)
done = False

for t in range(T):
    curdone = True
    for contestant in contestants:
        contestant.sim.add_transactions(workload_arr[t])
        if contestant.sim.model is not None:
            if contestant.name == "Linear Model":
                contestant.sim.sim(retryOnAbort=True, n=50, T=6, ML_RL="ML")
            elif contestant.name == "RL Model":
                contestant.sim.sim(retryOnAbort=True, n=50, T=6, ML_RL="RL")
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
        if contestant.name == "Linear Model":
            contestant.sim.sim(retryOnAbort=True, n=50, T=6, ML_RL="ML")
        elif contestant.name == "RL Model":
            contestant.sim.sim(retryOnAbort=True, n=50, T=6, ML_RL="RL")
        #print(contestant.sim.online_stats())
        d = contestant.sim.print_statistics()
        d["name"] = contestant.name
        curdone = curdone and contestant.sim.done()
        with open(filename, "a") as f:
            f.write(str(d) + "\n")
    done = done or curdone

# {'n_aborts': 0, 'n_successes': 10000, 'steps': 6369}
# {'n_aborts': 0, 'n_successes': 10000, 'steps': 6280}
