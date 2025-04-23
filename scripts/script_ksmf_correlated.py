from Simulator import Simulator
from Scheduler import KSMFScheduler
from Workload import SmallBankWorkload

corr_params = {
    "normal_scale": 10.0,
    "event_scale": 20.0,
    "correlation_factor": 0.1
}
workload = SmallBankWorkload(correlated=True, corr_params=corr_params)
probabilities = [0.15, 0.15, 0.15, 0.25, 0.15, 0.15] # https://github.com/cmu-db/benchbase/blob/main/config/mysql/sample_smallbank_config.xml#L22

scheduler = KSMFScheduler(k=5)
sim = Simulator(scheduler, [])

T = 100

for t in range(T):
    num_txns = 100
    random_transactions = workload.generate_transactions(num_txns, probabilities, start=t*100)
    sim.add_transactions(random_transactions)
    sim.sim()

# clean up the remaining
while not sim.done():
    sim.sim()

sim.print_statistics()

