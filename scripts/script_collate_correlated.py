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
workload = SmallBankWorkload(correlated=True, corr_params=corr_params)
# workload = TPCCWorkload(correlated=False, corr_params=corr_params)
probabilities = [0.15, 0.15, 0.15, 0.25, 0.15, 0.15] # https://github.com/cmu-db/benchbase/blob/main/config/mysql/sample_smallbank_config.xml#L22
# probabilities = [43,41,3,1,3,1,3,1,4]
workload_arr = []
oracle_arr = []
T = 1
filename = "results/arena_corr_SB_2.txt"
timing_filename = "results/arena_corr_SB_times_2.txt"
for t in range(T):
    num_txns = 1000
    random_transactions = workload.generate_transactions(num_txns, probabilities=probabilities, start=t*num_txns)
    oracle_arr.append(workload.get_sticky_list())
    workload_arr.append([clone_transaction(txn) for txn in random_transactions])

class Contestant():
    def __init__(self, name:str, scheduler:Scheduler, model=None, n:int=None, T:int=None, filterT:bool = False, skipT:bool = False):
        self.name = name
        self.scheduler = scheduler
        self.sim = Simulator(scheduler, [], filterT, skipT)
        self.sim.model = model
        self.done = False
        self.makespan = None
        self.n = n
        self.T = T


contestants:list[Contestant] = []

# with open('model_linear.pkl', 'rb') as f:
#    model_linear = pickle.load(f)

with open('model_RL_20_TPCC_corr.pkl', 'rb') as f:
    model_RL_20_TPCC_corr = pickle.load(f)
with open('model_RL_50_TPCC_corr.pkl', 'rb') as f:
    model_RL_50_TPCC_corr = pickle.load(f)
with open('model_RL_100_TPCC_corr.pkl', 'rb') as f:
    model_RL_100_TPCC_corr = pickle.load(f)

with open('model_RL_20_SB_corr.pkl', 'rb') as f:
    model_RL_20_SB_corr = pickle.load(f)
with open('model_RL_50_SB_corr.pkl', 'rb') as f:
    model_RL_50_SB_corr = pickle.load(f)
with open('model_RL_100_SB_corr.pkl', 'rb') as f:
    model_RL_100_SB_corr = pickle.load(f)


# with open('model_RL_20_TPCC_corr.pkl', 'rb') as f:
    # model_RL_20 = pickle.load(f)


contestants.extend(
    [
        Contestant("k-smf-k=5", scheduler = KSMFScheduler(k=5)),
        Contestant("k-smf-twisted-k=5", scheduler = KSMFTwistedScheduler(k=5)),
        # # Contestant("k-smf-twisted-k=10", scheduler = KSMFTwistedScheduler(k=10)),
        # # Contestant("k-smf-k=10", scheduler = KSMFScheduler(k=10)),
        # Contestant("k-smf-oracle-k=5", scheduler = KSMFOracleScheduler(k=5)),
        # #Contestant("k-smf-oracle-2-phase", scheduler = KSMFOracle2PhaseScheduler(k=5)),
        # Contestant("k-smf-oracle-2-phase-k=5", scheduler = KSMFOracle2PhaseDontCareScheduler(k=5)),
        Contestant("k-smf-twisted-oracle-2-phase-k=5", scheduler = KSMFTwistedOracle2PhaseDontCareScheduler(k=5)),
        # Contestant("int-opt-k=10 n=20", QueueKernelScheduler(n_queues=10, kernel=KernelWrapper(IntegerOptimisationKernelMkII(20)))),
        #Contestant("k-smf-zero", scheduler = KSMFZeroScheduler(k=5))
        # Contestant("Linear Model", scheduler=LumpScheduler(), model=model_linear),
        # Contestant("RL-ordered k-smf n=50 T=7", scheduler=RLSMFTwistedScheduler(), model=model_RL, filterT = False),
        # Contestant("RL-ordered k-smf n=50 T=7", scheduler=RLSMFTwistedScheduler(), model=model_RL, filterT = True),

        Contestant("RL n=20 T=6", scheduler=LumpScheduler(), model=model_RL_20_SB_corr, n=20, T=6, filterT = False, skipT = False),
        Contestant("RL n=20 T=6 filterT", scheduler=LumpScheduler(), model=model_RL_20_SB_corr, n=20, T=6, filterT = True, skipT = False),
        Contestant("RL n=20 T=6 skipT", scheduler=LumpScheduler(), model=model_RL_20_SB_corr, n=20, T=6, filterT = False, skipT = True),
        Contestant("RL n=20 T=6 filterT skipT", scheduler=LumpScheduler(), model=model_RL_20_SB_corr, n=20, T=6, filterT = True, skipT = True),
        Contestant("RL-smf n=20 T=6", scheduler=RLSMFTwistedScheduler(), model=model_RL_20_SB_corr, n=20, T=6, filterT = False, skipT = False),
        # Contestant("RL-smf n=20 T=6 filterT", scheduler=RLSMFTwistedScheduler(), model=model_RL_20_SB_corr, n=20, T=6, filterT = True, skipT = False),

        Contestant("RL n=50 T=6", scheduler=LumpScheduler(), model=model_RL_50_SB_corr, n=50, T=6, filterT = False, skipT = False),
        Contestant("RL n=50 T=6 filterT", scheduler=LumpScheduler(), model=model_RL_50_SB_corr, n=50, T=6, filterT = True, skipT = False),
        Contestant("RL n=50 T=6 skipT", scheduler=LumpScheduler(), model=model_RL_50_SB_corr, n=50, T=6, filterT = False, skipT = True),
        Contestant("RL n=50 T=6 filterT skipT", scheduler=LumpScheduler(), model=model_RL_50_SB_corr, n=50, T=6, filterT = True, skipT = True),
        Contestant("RL-smf n=50 T=6", scheduler=RLSMFTwistedScheduler(), model=model_RL_50_SB_corr, n=50, T=6, filterT = False, skipT = False),
        # Contestant("RL-smf n=50 T=6 filterT", scheduler=RLSMFTwistedScheduler(), model=model_RL_50_SB_corr, n=50, T=6, filterT = True, skipT = False),

        Contestant("RL n=100 T=6", scheduler=LumpScheduler(), model=model_RL_100_SB_corr, n=100, T=6, filterT = False, skipT = False),
        Contestant("RL n=100 T=6 filterT", scheduler=LumpScheduler(), model=model_RL_100_SB_corr, n=100, T=6, filterT = True, skipT = False),
        Contestant("RL n=100 T=6 skipT", scheduler=LumpScheduler(), model=model_RL_100_SB_corr, n=100, T=6, filterT = False, skipT = True),
        Contestant("RL n=100 T=6 filterT skipT", scheduler=LumpScheduler(), model=model_RL_100_SB_corr, n=100, T=6, filterT = True, skipT = True),
        Contestant("RL-smf n=100 T=6", scheduler=RLSMFTwistedScheduler(), model=model_RL_100_SB_corr, n=100, T=6, filterT = False, skipT = False),
        # Contestant("RL-smf n=100 T=6 filterT", scheduler=RLSMFTwistedScheduler(), model=model_RL_100_SB_corr, n=100, T=6, filterT = True, skipT = False),
        

        # Contestant("RL n=20 T=7", scheduler=LumpScheduler(), model=model_RL_20_TPCC_corr, n=20, T=7, filterT = False, skipT = False),
        # Contestant("RL n=20 T=7 filterT", scheduler=LumpScheduler(), model=model_RL_20_TPCC_corr, n=20, T=7, filterT = True, skipT = False),
        # Contestant("RL-smf n=20 T=7", scheduler=RLSMFTwistedScheduler(), model=model_RL_20_TPCC_corr, n=20, T=7, filterT = False, skipT = False),
        # Contestant("RL-smf n=20 T=7 filterT", scheduler=RLSMFTwistedScheduler(), model=model_RL_20_TPCC_corr, n=20, T=7, filterT = True, skipT = False),

        # Contestant("RL n=50 T=7", scheduler=LumpScheduler(), model=model_RL_50_TPCC_corr, n=50, T=7, filterT = False, skipT = False),
        # Contestant("RL n=50 T=7 filterT", scheduler=LumpScheduler(), model=model_RL_50_TPCC_corr, n=50, T=7, filterT = True, skipT = False),
        # Contestant("RL-smf n=50 T=7", scheduler=RLSMFTwistedScheduler(), model=model_RL_50_TPCC_corr, n=50, T=7, filterT = False, skipT = False),
        # Contestant("RL-smf n=50 T=7 filterT", scheduler=RLSMFTwistedScheduler(), model=model_RL_50_TPCC_corr, n=50, T=7, filterT = True, skipT = False),

        # Contestant("RL n=100 T=7", scheduler=LumpScheduler(), model=model_RL_100_TPCC_corr, n=100, T=7, filterT = False, skipT = False),
        # Contestant("RL n=100 T=7 filterT", scheduler=LumpScheduler(), model=model_RL_100_TPCC_corr, n=100, T=7, filterT = True, skipT = False),
        # Contestant("RL-smf n=100 T=7", scheduler=RLSMFTwistedScheduler(), model=model_RL_100_TPCC_corr, n=100, T=7, filterT = False, skipT = False),
        # Contestant("RL-smf n=100 T=7 filterT", scheduler=RLSMFTwistedScheduler(), model=model_RL_100_TPCC_corr, n=100, T=7, filterT = True, skipT = False),
       
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
            contestant.sim.sim(retryOnAbort=True, n=contestant.n, T=contestant.T, ML_RL="RL")
        else:
            contestant.sim.sim(retryOnAbort=True)
        end_time = time.time()
        # print('total pass time', str((end_time - start_time) * 1000)[:5], flush=True)
        
        d = contestant.sim.print_statistics()
        ts = int((end_time - start_time) * 1000)
        print(d, flush=True)
        times = {
            "name": contestant.name,
            "total_time": ts,
            "decision_time": d.get("decision_time"),
            "conflict_time": d.get("conflict_time"),
            "RL_time": d.get("RL_time"),
        }
        d["name"] = contestant.name
        with open(filename, "a") as f:
            f.write(str(d) + "\n")
        with open(timing_filename, "a") as f:
            f.write(str(times) + "\n")
        contestant.sim.reset_ts_stats()
    done = done or curdone

# clean up the remaining
done_counter = len(contestants)
while done_counter > 0:
    for contestant in contestants:
        if contestant.sim.finished == True: 
            if contestant.done is False:
                contestant.done = True
                contestant.makespan = contestant.sim.step
                print(f'contestant {contestant.name} is done, makespan = {contestant.makespan}', flush=True)
                done_counter -= 1
                continue
        else:
            start_time = time.time()
            if contestant.sim.model is not None:
                contestant.sim.sim(retryOnAbort=True, n=contestant.n, T=contestant.T, ML_RL="RL")
            else:
                contestant.sim.sim(retryOnAbort=True)
            end_time = time.time()
            # print('total pass time', str((end_time - start_time) * 1000)[:5], flush=True)
            
            d = contestant.sim.print_statistics()
            ts = int((end_time - start_time) * 1000)
            times = {
                "name": contestant.name,
                "total_time": ts,
                "decision_time": d.get("decision_time"),
                "conflict_time": d.get("conflict_time"),
                "RL_time": d.get("RL_time"),
            }
            d["name"] = contestant.name
            with open(filename, "a") as f:
                f.write(str(d) + "\n")
            with open(timing_filename, "a") as f:
                f.write(str(times) + "\n")
            contestant.sim.reset_ts_stats()

for contestant in contestants:
    d = contestant.sim.statistics
    d["name"] = contestant.name
    print(d)

# {'n_aborts': 0, 'n_successes': 10000, 'steps': 6369}
# {'n_aborts': 0, 'n_successes': 10000, 'steps': 6280}
