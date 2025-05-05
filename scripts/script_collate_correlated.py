from Simulator import Simulator
from Scheduler import Scheduler, QueueKernelScheduler, KSMFScheduler, QueueScheduler, KSMFOracleScheduler, KSMFOracle2PhaseScheduler, KSMFZeroScheduler, KSMFOracle2PhaseDontCareScheduler, KSMFTwistedOracle2PhaseDontCareScheduler, KSMFTwistedScheduler
from Workload import SmallBankWorkload
from KernelWrapper import KernelWrapper
from Kernel import IntegerOptimisationKernelMkII
from utils import clone_transaction

corr_params = {
    "normal_scale": 10.0,
    "event_scale": 20.0,
    "correlation_factor": 0.1
}
workload = SmallBankWorkload(correlated=True, corr_params=corr_params)
probabilities = [0.15, 0.15, 0.15, 0.25, 0.15, 0.15] # https://github.com/cmu-db/benchbase/blob/main/config/mysql/sample_smallbank_config.xml#L22

workload_arr = []
oracle_arr = []
T = 100
filename = "arena_correlated_twisted.txt"
for t in range(T):
    num_txns = 1000
    random_transactions = workload.generate_transactions(num_txns, probabilities, start=t*num_txns)
    oracle_arr.append(workload.get_sticky_list())
    workload_arr.append([clone_transaction(txn) for txn in random_transactions])

class Contestant():
    def __init__(self, name:str, scheduler:Scheduler):
        self.name = name
        self.scheduler = scheduler
        self.sim = Simulator(scheduler, [])

contestants:list[Contestant] = []

contestants.extend(
    [
        Contestant("k-smf-k=5", scheduler = KSMFScheduler(k=5)),
        Contestant("k-smf-twisted-k=5", scheduler = KSMFTwistedScheduler(k=5)),
        Contestant("k-smf-twisted-k=10", scheduler = KSMFTwistedScheduler(k=10)),
        Contestant("k-smf-k=10", scheduler = KSMFScheduler(k=10)),
        Contestant("k-smf-oracle-k=5", scheduler = KSMFOracleScheduler(k=5)),
        #Contestant("k-smf-oracle-2-phase", scheduler = KSMFOracle2PhaseScheduler(k=5)),
        Contestant("k-smf-oracle-2-phase-k=5", scheduler = KSMFOracle2PhaseDontCareScheduler(k=5)),
        Contestant("k-smf-twisted-k=5", scheduler = KSMFTwistedOracle2PhaseDontCareScheduler(k=5))
        #Contestant("k-smf-zero", scheduler = KSMFZeroScheduler(k=5))
    ]
)

done = False

for t in range(T):
    curdone = True
    for contestant in contestants:
        contestant.sim.add_transactions(workload_arr[t])
        if isinstance(contestant.sim.scheduler, KSMFOracleScheduler) or isinstance(contestant.sim.scheduler, KSMFOracle2PhaseScheduler) or isinstance(contestant.sim.scheduler, KSMFOracle2PhaseDontCareScheduler):
            contestant.sim.scheduler.inject_oracle_list(oracle_arr[t])
        contestant.sim.sim(retryOnAbort=True)
        curdone = curdone and contestant.sim.done()
    done = done or curdone

# clean up the remaining
while not done:
    curdone = True
    for contestant in contestants:
        if contestant.sim.done(): continue
        contestant.sim.sim(retryOnAbort=True)
        curdone = curdone and contestant.sim.done()
    done = done or curdone

for contestant in contestants:
    d = contestant.sim.statistics
    d["name"] = contestant.name
    print(d)

# {'n_aborts': 0, 'n_successes': 10000, 'steps': 6369}
# {'n_aborts': 0, 'n_successes': 10000, 'steps': 6280}
