from Simulator import Simulator
from Scheduler import QueueBasedScheduler
from Workload import SmallBankWorkload

workload = SmallBankWorkload()
probabilities = [0.15, 0.15, 0.15, 0.25, 0.15, 0.15] # https://github.com/cmu-db/benchbase/blob/main/config/mysql/sample_smallbank_config.xml#L22
num_txns = 1000
random_transactions = workload.generate_random_transactions(num_txns, probabilities)

scheduler = QueueBasedScheduler(10)
sim = Simulator(scheduler, random_transactions)

while not sim.done():
    sim.sim()
sim.print_statistics()
