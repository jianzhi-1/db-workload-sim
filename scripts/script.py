import argparse
from Simulator import Simulator
from Scheduler import SequentialScheduler
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
    probabilities = [43,41,3,1,3,1,3,1,4]  # https://github.com/cmu-db/benchbase/blob/main/config/mysql/sample_tpcc_config.xml

num_txns = 100
random_transactions = workload.generate_random_transactions(num_txns, probabilities)

scheduler = SequentialScheduler()
sim = Simulator(scheduler, random_transactions)

while not sim.done():
    sim.sim()
sim.print_statistics()
