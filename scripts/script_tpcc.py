from Simulator import Simulator
from Scheduler import SequentialScheduler, KSMFScheduler
from Workload import TPCCWorkload

workload = TPCCWorkload()
probabilities = [43,41,3,1,3,1,3,1,4] # https://github.com/cmu-db/benchbase/blob/main/config/mysql/sample_tpcc_config.xml
num_txns = 1000
random_transactions = workload.generate_random_transactions(num_txns, probabilities)
print(random_transactions)

scheduler = KSMFScheduler(k=5) #SequentialScheduler()
sim = Simulator(scheduler, random_transactions)

while not sim.done():
    sim.sim()
sim.print_statistics()