import argparse
from Simulator import Simulator
from Scheduler import KSMFScheduler
from Workload import SmallBankWorkload, TPCCWorkload

corr_params = {
    "normal_scale": 10.0,
    "event_scale": 20.0,
    "correlation_factor": 0.1
}

parser = argparse.ArgumentParser()
parser.add_argument("--workload", choices=["smallbank", "tpcc"], default="smallbank", help="Workload type to use")
args = parser.parse_args()

# Initialize workload based on parameter
if args.workload == "smallbank":
    workload = SmallBankWorkload(correlated=True, corr_params=corr_params)
    probabilities = [0.15, 0.15, 0.15, 0.25, 0.15, 0.15] # https://github.com/cmu-db/benchbase/blob/main/config/mysql/sample_smallbank_config.xml#L22
else:  # tpcc
    workload = TPCCWorkload(correlated=True, corr_params=corr_params)
    probabilities = [43,41,3,1,3,1,3,1,4]  # https://github.com/cmu-db/benchbase/blob/main/config/mysql/sample_tpcc_config.xml

scheduler = KSMFScheduler(k=5)
sim = Simulator(scheduler, [])

T = 100

for t in range(T):
    num_txns = 1000
    random_transactions = workload.generate_transactions(num_txns, probabilities, start=t*num_txns)
    sim.add_transactions(random_transactions)
    sim.sim()

# clean up the remaining
while not sim.done():
    sim.sim()

sim.print_statistics()

