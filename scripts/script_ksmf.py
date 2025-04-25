import argparse
from Simulator import Simulator
from Scheduler import KSMFScheduler
from Workload import SmallBankWorkload, TPCCWorkload

parser = argparse.ArgumentParser()
parser.add_argument("--workload", choices=["smallbank", "tpcc"], default="smallbank", help="Workload type to use")
args = parser.parse_args()

# Initialize workload based on parameter
if args.workload == "smallbank":
    workload = SmallBankWorkload()
    probabilities = [0.15, 0.15, 0.15, 0.25, 0.15, 0.15] # https://github.com/cmu-db/benchbase/blob/main/config/mysql/sample_smallbank_config.xml#L22
else:  # tpcc
    workload = TPCCWorkload()
    probabilities = None  # TPCC uses uniform distribution by default

scheduler = KSMFScheduler(k=5)
sim = Simulator(scheduler, [])

T = 100

for t in range(T):
    num_txns = 1000
    random_transactions = workload.generate_random_transactions(num_txns, probabilities, start=t*num_txns)
    sim.add_transactions(random_transactions)
    sim.sim()

# clean up the remaining
while not sim.done():
    sim.sim()

# Return statistics instead of printing
print(sim.result_statistics)

# {'n_aborts': 0, 'n_successes': 10000, 'steps': 6369}
# {'n_aborts': 0, 'n_successes': 10000, 'steps': 6280}
