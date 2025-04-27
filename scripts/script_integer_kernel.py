from Simulator import Simulator
from Scheduler import QueueKernelScheduler
from Workload import SmallBankWorkload
from KernelWrapper import KernelWrapper
from Kernel import IntegerOptimisationKernelMkII

workload = SmallBankWorkload()
probabilities = [0.15, 0.15, 0.15, 0.25, 0.15, 0.15] # https://github.com/cmu-db/benchbase/blob/main/config/mysql/sample_smallbank_config.xml#L22

n_queues = 50
max_n_kernel = 50
scheduler = QueueKernelScheduler(n_queues=n_queues, kernel=KernelWrapper(IntegerOptimisationKernelMkII(max_n_kernel)))
sim = Simulator(scheduler, [])

T = 100

for t in range(T):
    num_txns = 10
    random_transactions = workload.generate_transactions(num_txns, probabilities, start=t*num_txns)
    sim.add_transactions(random_transactions)
    print("DONE")
    sim.sim(retryOnAbort=True)
    print(sim.online_stats())
    #logging.info('This message will be logged to the file.')
    sim.print_statistics()
    with open("integer_kernel.txt", "a") as f:
        f.write(str(sim.print_statistics()) + "\n")

# clean up the remaining
while not sim.done():
    sim.sim(retryOnAbort=True)
    with open("integer_kernel.txt", "a") as f:
        f.write(str(sim.print_statistics()) + "\n")

sim.print_statistics()

# {'n_aborts': 0, 'n_successes': 10000, 'steps': 6369}
# {'n_aborts': 0, 'n_successes': 10000, 'steps': 6280}
