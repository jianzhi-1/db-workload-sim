import pickle
import time
from Simulator import Simulator
from Scheduler import LumpScheduler, RLSMFTwistedScheduler, Scheduler, QueueKernelScheduler, KSMFScheduler, QueueScheduler, KSMFOracleScheduler, KSMFOracle2PhaseScheduler, KSMFZeroScheduler, KSMFOracle2PhaseDontCareScheduler, KSMFTwistedOracle2PhaseDontCareScheduler, KSMFTwistedScheduler
from Workload import SmallBankWorkload, TPCCWorkload
from KernelWrapper import KernelWrapper
from Kernel import IntegerOptimisationKernelMkII
from utils import clone_transaction

corr_params = {
    "normal_scale": 10.0,
    "event_scale": 20.0,
    "correlation_factor": 0.1
}
# workload = SmallBankWorkload(correlated=True, corr_params=corr_params)
workload = TPCCWorkload()
# probabilities = [0.15, 0.15, 0.15, 0.25, 0.15, 0.15] # https://github.com/cmu-db/benchbase/blob/main/config/mysql/sample_smallbank_config.xml#L22
probabilities = [43,41,3,1,3,1,3,1,4]
workload_arr = []
oracle_arr = []
T = 1
filename = "arena_corr_TPCC.txt"
timing_filename = "arena_corr_TPCC_times.txt"
for t in range(T):
    num_txns = 500
    random_transactions = workload.generate_transactions(num_txns, probabilities=probabilities, start=t*num_txns)
    oracle_arr.append(workload.get_sticky_list())
    workload_arr.append([clone_transaction(txn) for txn in random_transactions])

class Contestant():
    def __init__(self, name:str, scheduler:Scheduler, model=None):
        self.name = name
        self.scheduler = scheduler
        self.sim = Simulator(scheduler, [], False)
        self.sim.model = model
        self.done = False
        self.makespan = None

contestants:list[Contestant] = []

# with open('model_linear.pkl', 'rb') as f:
#    model_linear = pickle.load(f)

with open('model_RL_50_TPCC_corr.pkl', 'rb') as f:
    model_RL = pickle.load(f)

with open('model_RL_20_TPCC_corr.pkl', 'rb') as f:
    model_RL_20 = pickle.load(f)


contestants.extend(
    [
        Contestant("k-smf-k=5", scheduler = KSMFScheduler(k=5)),
        Contestant("k-smf-twisted-k=5", scheduler = KSMFTwistedScheduler(k=5)),
        # Contestant("k-smf-twisted-k=10", scheduler = KSMFTwistedScheduler(k=10)),
        # Contestant("k-smf-k=10", scheduler = KSMFScheduler(k=10)),
        Contestant("k-smf-oracle-k=5", scheduler = KSMFOracleScheduler(k=5)),
        #Contestant("k-smf-oracle-2-phase", scheduler = KSMFOracle2PhaseScheduler(k=5)),
        Contestant("k-smf-oracle-2-phase-k=5", scheduler = KSMFOracle2PhaseDontCareScheduler(k=5)),
        Contestant("k-smf-twisted-oracle-2-phase-k=5", scheduler = KSMFTwistedOracle2PhaseDontCareScheduler(k=5)),
        # Contestant("int-opt-k=10 n=20", QueueKernelScheduler(n_queues=10, kernel=KernelWrapper(IntegerOptimisationKernelMkII(20)))),
        #Contestant("k-smf-zero", scheduler = KSMFZeroScheduler(k=5))
        # Contestant("Linear Model", scheduler=LumpScheduler(), model=model_linear),
        Contestant("RL-ordered k-smf n=50 T=7", scheduler=RLSMFTwistedScheduler(), model=model_RL),
        Contestant("RL-ordered k-smf n=20 T=7", scheduler=RLSMFTwistedScheduler(), model=model_RL_20),
    ]
)

done = False

for t in range(T):
    curdone = True
    for contestant in contestants:
        contestant.sim.add_transactions(workload_arr[t])
        if isinstance(contestant.sim.scheduler, KSMFOracleScheduler) or isinstance(contestant.sim.scheduler, KSMFOracle2PhaseScheduler) or isinstance(contestant.sim.scheduler, KSMFOracle2PhaseDontCareScheduler):
            contestant.sim.scheduler.inject_oracle_list(oracle_arr[t])
        
        start_time = time.time()
        if contestant.sim.model is not None:
            if contestant.name == "Linear Model":
                contestant.sim.sim(retryOnAbort=True, n=50, T=7, ML_RL="ML")
            elif contestant.name == "RL-ordered k-smf n=50 T=7":
                contestant.sim.sim(retryOnAbort=True, n=50, T=7, ML_RL="RL")
            elif contestant.name == "RL-ordered k-smf n=20 T=7":
                contestant.sim.sim(retryOnAbort=True, n=20, T=7, ML_RL="RL")
        else:
            contestant.sim.sim(retryOnAbort=True)
        end_time = time.time()
        
        d = contestant.sim.print_statistics()
        ts = int((end_time - start_time) * 10000)
        times = {
            "name": contestant.name,
            "time": ts,
        }
        d["name"] = contestant.name
        with open(filename, "a") as f:
            f.write(str(d) + "\n")
        with open(timing_filename, "a") as f:
            f.write(str(times) + "\n")
    done = done or curdone

# clean up the remaining
done_counter = len(contestants)
while done_counter > 0:
    for contestant in contestants:
        if contestant.sim.done(): 
            if contestant.done is False:
                contestant.done = True
                contestant.makespan = contestant.sim.step
                done_counter -= 1
                continue
        else:
            start_time = time.time()
            if contestant.sim.model is not None:
                if contestant.name == "Linear Model":
                    contestant.sim.sim(retryOnAbort=True, n=50, T=7, ML_RL="ML")
                elif contestant.name == "RL-ordered k-smf n=50 T=7":
                    contestant.sim.sim(retryOnAbort=True, n=50, T=7, ML_RL="RL")
                elif contestant.name == "RL-ordered k-smf n=20 T=7":
                    contestant.sim.sim(retryOnAbort=True, n=20, T=7, ML_RL="RL")
            else:
                contestant.sim.sim(retryOnAbort=True)
            end_time = time.time()
            
            d = contestant.sim.print_statistics()
            ts = int((end_time - start_time) * 10000)
            times = {
                "name": contestant.name,
                "time": ts,
            }
            d["name"] = contestant.name
            with open(filename, "a") as f:
                f.write(str(d) + "\n")
            with open(timing_filename, "a") as f:
                f.write(str(times) + "\n")

for contestant in contestants:
    d = contestant.sim.statistics
    d["name"] = contestant.name
    print(d)

# {'n_aborts': 0, 'n_successes': 10000, 'steps': 6369}
# {'n_aborts': 0, 'n_successes': 10000, 'steps': 6280}
