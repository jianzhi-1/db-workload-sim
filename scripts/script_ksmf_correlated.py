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
filename = "ksmf_correlated.txt"

for t in range(T):
    num_txns = 1000
    random_transactions = workload.generate_transactions(num_txns, probabilities, start=t*num_txns)
    sim.add_transactions(random_transactions)
    sim.sim()
    with open(filename, "a") as f:
        f.write(str(sim.print_statistics()) + "\n")

# clean up the remaining
while not sim.done():
    sim.sim()
    with open(filename, "a") as f:
        f.write(str(sim.print_statistics()) + "\n")

sim.print_statistics()

