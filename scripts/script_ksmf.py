from Simulator import Simulator
from Scheduler import KSMFScheduler
from Workload import SmallBankWorkload

workload = SmallBankWorkload()
probabilities = [0.15, 0.15, 0.15, 0.25, 0.15, 0.15] # https://github.com/cmu-db/benchbase/blob/main/config/mysql/sample_smallbank_config.xml#L22

scheduler = KSMFScheduler(k=5)
sim = Simulator(scheduler, [])

T = 100

for t in range(T):
    num_txns = 100
    random_transactions = workload.generate_random_transactions(num_txns, probabilities, start=t*100)
    sim.add_transactions(random_transactions)
    sim.sim()

# clean up the remaining
while not sim.done():
    sim.sim()

sim.print_statistics()

# {'n_aborts': 203, 'n_successes': 9797, 'steps': 14330}
