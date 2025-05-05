import heapq
from Scheduler import Scheduler
from utils import Transaction, Operation, ReadOperation, WriteOperation
from Locker import Locker
from utils import clone_transaction, clone_operation_list, conflict
import torch
import numpy as np

DELTA = 10000000 # must be greater than total number of transactions
# determines how much to increment transaction number on reschedule

def conflict_matrix(batch_size, num_txns, T, random_transactions):
    arr = np.zeros((batch_size, num_txns, num_txns, 2*T + 1))
    for t in range(-T, T + 1, 1):
        for i, t1 in enumerate(random_transactions):
            for j, t2 in enumerate(random_transactions):
                arr[0][i][j][t + T] = conflict(t1, t2, t)
    return arr

class Pair():
    def __init__(self, priority:int, txn:Transaction):
        self.priority = priority
        self.txn = txn
    
    def __lt__(self, other):
        return self.priority < other.priority

class Simulator():

    ### Assumptions
    # 1. Read and write operations take the same time (1 tick)
    # 2. If T2 requests for a resource that T1 has not released, then T2 aborts itself and n_aborts += 1. 
    #    T2 does not restart itself (for simplicity of simulation, otherwise just implementing MVCC)

    def __init__(self, scheduler:Scheduler=None, txnPool:list[Transaction]=None):
        self.scheduler = scheduler
        self.txnPool = txnPool
        self.resource_locks = Locker()
        self.graveyard = set()
        self.inflight = dict()
        self.scheduled_txn = []
        self.scheduled_time = dict()
        self.step = 0
        self.memo = dict()
        self.clear()
        self.flushPool = [] # flush for Oracle
        self.model = None

    def clear(self):
        self.resource_locks = Locker()
        self.graveyard = set()
        self.inflight = dict()
        self.scheduled_time = dict()
        self.step = 0

        # statistics
        self.statistics = {
            "n_aborts": 0,
            "n_successes": 0,
            "steps": 0
        }

        self.result_statistics = dict()

    def add_transactions(self, more_txns:list[Transaction]):
        # add more transactions to transaction pool, for possibly online use cases
        self.txnPool.extend([clone_transaction(txn) for txn in more_txns]) # clone just to be safe

    def flush(self):
        # flush all scheduled transaction back into transaction pool
        self.flushPool = []
        while len(self.scheduled_txn) > 0:
            p = heapq.heappop(self.scheduled_txn)
            new_txn = clone_transaction(p.txn)
            new_txn.txn += DELTA
            self.flushPool.append(new_txn)

    def sim(self, freeze:bool=False, retryOnAbort:bool=False, n:int=None, T:int=None) -> dict:
        while len(self.scheduled_txn) > 0 and self.scheduled_txn[0].priority <= self.step:
            if self.scheduled_txn[0].priority < self.step: assert False, "transaction should be scheduled earlier"
            p = heapq.heappop(self.scheduled_txn)
            self.inflight[p.txn.txn] = p.txn.operations
            self.memo[p.txn.txn] = clone_transaction(p.txn) # clone just in case need to reschedule

        if len(self.txnPool) == 0: # no more transactions to be scheduled
            self.tick(retryOnAbort=retryOnAbort)
            return self.statistics
        
        txns_to_schedule:list[Transaction] = self.txnPool
        if n != None and T != None:
            txns_to_schedule = self.txnPool[:n]
            #if len(txns_to_schedule != n):
            #    txns_to_schedule.extend([Transaction(-1, [], "") for _ in range(n - len(txns_to_schedule))])

            x = conflict_matrix(1, n, T, txns_to_schedule)
            x = torch.from_numpy(x.astype(np.float32))
            lamb, p = self.model(x)
            memory = dict()
            for j, txn in enumerate(txns_to_schedule):
                prob_arr = lamb[0][j].cpu().detach().numpy()
                memory[txn.txn] = np.random.choice(range(self.step, self.step+T+2), p=prob_arr, size=1)[0]
            self.scheduler.memory = memory

        decisions = self.scheduler.schedule(self.inflight, self, txns_to_schedule, self.step) # pass self for flush()
        if (n != None and len(self.txnPool) > len(txns_to_schedule)):
            decisions.extend([0]*(len(self.txnPool) - len(txns_to_schedule)))

        assert len(decisions) == len(self.txnPool), "Decision length is not equal transaction pool length"

        new_pool = []
        for i, t in enumerate(self.txnPool):
            if decisions[i] >= 1: 
                self.scheduled_time[t.txn] = self.step + decisions[i] - 1
                if decisions[i] == 1:
                    self.inflight[t.txn] = t.operations
                    self.memo[t.txn] = clone_transaction(t)
                else:
                    heapq.heappush(self.scheduled_txn, Pair(self.scheduled_time[t.txn], t))
            elif decisions[i] == -1:
                pass # scheduler says toss this transaction away
            elif decisions[i] == 0: new_pool.append(t)
            else: assert False, f"unknown decision {decisions[i]}"
        self.txnPool = new_pool + self.flushPool
        self.flushPool = []

        statistics = dict()

        if freeze: statistics |= self.sim_frozen()
        self.tick(retryOnAbort=retryOnAbort) # tick one step

        return (self.statistics | statistics) if freeze else self.statistics

    def sim_frozen(self) -> dict:

        # freeze the simulation at the current time step and execute all the transactions

        locker_clone = Locker(self.resource_locks)
        inflight_clone = {k:clone_operation_list(self.inflight[k]) for k in self.inflight}
        statistics = {
            "freeze_aborts": 0,
            "freeze_successes": 0,
            "freeze_steps": 0
        }

        while len(inflight_clone) > 0:

            inflight_keyset = list(inflight_clone.keys())
            
            for txn in inflight_keyset:
                if len(inflight_clone[txn]) == 0: assert False, "empty transaction for some reason"
                op = inflight_clone[txn][0] # get the first operation
                can = locker_clone.lock(op.resource, op.txn, op.type, self.step + op.delta_last + 1)
                if not can: # conflict
                    statistics["freeze_aborts"] += 1
                    locker_clone.remove_all(op.txn)
                    inflight_clone[txn] = []
                else: # success
                    inflight_clone[txn] = inflight_clone[txn][1:] # pop the first operations
                    if op.is_last_on_resource:
                        locker_clone.remove(op.resource, op.txn)

                    if op.is_last:
                        locker_clone.remove_all(op.txn) # not necessary, but for peace of mind
                        statistics["freeze_successes"] += 1

                if len(inflight_clone[txn]) == 0:
                    del inflight_clone[txn]
                
            statistics["freeze_steps"] += 1

        return statistics

    def tick(self, retryOnAbort:bool=False) -> dict:

        inflight_keyset = list(self.inflight.keys())
        
        for txn in inflight_keyset:
            if len(self.inflight[txn]) == 0: assert False, "empty transaction for some reason"
            op = self.inflight[txn][0] # get the first operation
            can = self.resource_locks.lock(op.resource, op.txn, op.type, self.step + op.delta_last + 1)
            if not can: # conflict
                self.result_statistics[txn] = 0
                self.statistics["n_aborts"] += 1
                self.resource_locks.remove_all(op.txn)
                self.inflight[txn] = []
                if retryOnAbort: 
                    new_txn = self.memo[txn]
                    # print(f"Rescheduled: {new_txn.txn} -> {new_txn.txn+DELTA}: {new_txn}")
                    new_txn.txn += DELTA
                    self.txnPool.append(clone_transaction(new_txn))
            else: # success
                self.inflight[txn] = self.inflight[txn][1:] # pop the first operations
                if op.is_last_on_resource:
                    self.resource_locks.remove(op.resource, op.txn)

                if op.is_last:
                    self.result_statistics[txn] = 1
                    self.resource_locks.remove_all(op.txn) # not necessary, but for peace of mind
                    self.statistics["n_successes"] += 1
                    del self.memo[txn] # no need to remember the original transaction anymore

            if len(self.inflight[txn]) == 0:
                del self.inflight[txn]

        self.statistics["steps"] += 1
        self.step += 1

    def done(self) -> bool:
        return len(self.inflight) == 0 and len(self.txnPool) == 0 and len(self.scheduled_txn) == 0
    
    def print_statistics(self) -> dict:
        print(self.statistics)
        return self.statistics

    def online_stats(self):
        return self.step, len(self.inflight) + len(self.txnPool) + len(self.scheduled_txn)

