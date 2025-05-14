import heapq
import time
import numpy as np
import torch
from Scheduler import LumpScheduler, Scheduler, KSMFTwistedScheduler, KSMFTwistedOracle2PhaseDontCareScheduler, RLSMFTwistedScheduler
from utils import Transaction, Operation, ReadOperation, WriteOperation, conflict
from Locker import Locker
from utils import clone_transaction, clone_operation_list

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

    def __init__(self, scheduler:Scheduler=None, txnPool:list[Transaction]=[], filterT:bool=False, skipT:bool = False):
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
        # self.txns_ML = []
        self.filterT:bool = filterT #filter out transactions that conflict with inflight transactions
        self.skipT:bool = skipT #make a scheduling decision every T timesteps instead of every 1
        self.finished = False

    def clear(self):
        self.resource_locks = Locker()
        self.graveyard = set()
        self.inflight = dict()
        self.scheduled_time = dict()
        self.step = 0
        self.finished = False

        # statistics
        self.statistics = {
            "n_aborts": 0,
            "n_successes": 0,
            "steps": 0
        }

        self.result_statistics = dict()

    def add_transactions(self, more_txns:list[Transaction]):
        # add more transactions to transaction pool, for possibly online use cases
        self.finished = False # just in case it was true
        self.txnPool.extend([clone_transaction(txn) for txn in more_txns]) # clone just to be safe

    def flush(self):
        # flush all scheduled transaction back into transaction pool
        self.flushPool = []
        while len(self.scheduled_txn) > 0:
            p = heapq.heappop(self.scheduled_txn)
            new_txn = clone_transaction(p.txn)
            new_txn.txn += DELTA
            self.flushPool.append(new_txn)

    def no_filter_T(self, n):
        # res = [clone_transaction(txn) for txn in self.txnPool[:n]]
        res = self.txnPool[:n]
        self.txnPool = self.txnPool[n:]
        return res
    
    def filter_T(self, n): #filtered set of transactions to schedule
        res = []
        num_candidates = 0
        idx = 0
        maxlim = len(self.txnPool)
        while num_candidates < n and idx < len(self.txnPool): #Filter based on whether there is a current conflict
            if idx > maxlim:
                assert False, "No valid resources left to schedule easily"
            candidate = self.txnPool[idx]
            can_schedule = True
            for op in candidate.operations:
                # typ = "R" if op.is_read else "W"
                # if self.locker.probe(op.resource, typ, self.step, T) < self.step:
                # print(self.resource_locks.probe_table, flush=True)
                # if op.resource in self.resource_locks.probe_table:
                #     if (self.resource_locks.probe_table[op.resource] and 
                #         (self.resource_locks.probe_table[op.resource][0] is not None) and 
                #         (self.resource_locks.probe_table[op.resource][1] is not None)):
                #         can_schedule = False
                #         break

                # if op.resource 
                if self.resource_locks.probe(op.resource, op.type) >= self.step:
                    can_schedule = False
                    break
            if can_schedule:
                res.append(clone_transaction(candidate))
                num_candidates += 1
            else:
                self.txnPool.append(clone_transaction(candidate))
            idx += 1
        self.txnPool = self.txnPool[idx:]
        return res


    
    def update_memory_ML(self, n, T):
        self.scheduler.memory = {}
        txns_to_schedule = []
        if self.filterT == False:
            txns_to_schedule = self.no_filter_T(n)
        else:
            txns_to_schedule = self.filter_T(n)
        x = conflict_matrix(1, n, T, txns_to_schedule)
        x = torch.from_numpy(x.astype(np.float32))
        lamb, p = self.model(x)
        # memory = dict()
        for j, txn in enumerate(txns_to_schedule):
            prob_arr = lamb[0][j].cpu().detach().numpy()
            action = np.random.choice(range(self.step, self.step+T+2), p=prob_arr, size=1)[0]
            if action != self.step+T+1:
                self.scheduler.memory[txn.txn] = action
        # self.scheduler.memory = memory
        return txns_to_schedule
    
    def update_memory_RL(self, n, T):
        txns_N = []
        if self.filterT == False: #removes txns to schedule from the pool, puts them back after
            txns_N = self.no_filter_T(n)
        else:
            txns_N = self.filter_T(n)

        # self.scheduler.memory = {}

        # self.txns_ML = []

        # num_candidates = 0
        # idx = 0
        # maxlim = len(self.txnPool)
        # while num_candidates < n and idx < len(self.txnPool): #Filter based on whether there is a current conflict
        #     if idx > maxlim:
        #         assert False, "No valid resources left to schedule easily"
        #     candidate = self.txnPool[idx]
        #     can_schedule = True
        #     for op in candidate.operations:
        #         # typ = "R" if op.is_read else "W"
        #         # if self.locker.probe(op.resource, typ, self.step, T) < self.step:
        #         # print(self.resource_locks.probe_table, flush=True)
        #         if op.resource in self.resource_locks.probe_table:
        #             if (self.resource_locks.probe_table[op.resource] and 
        #                 (self.resource_locks.probe_table[op.resource][0] is not None) and 
        #                 (self.resource_locks.probe_table[op.resource][1] is not None)):
        #                 can_schedule = False
        #                 break
        #     if can_schedule:
        #         self.txns_ML.append(clone_transaction(candidate))
        #         num_candidates += 1
        #     else:
        #         self.txnPool.append(clone_transaction(candidate))
        #     idx += 1
        
        # self.txnPool = self.txnPool[idx:]

        txns_to_schedule = []

        start_time = time.time()
        x = conflict_matrix(1, n, T, txns_N)
        end_time = time.time()
        # print('conflict_matrix', str((end_time - start_time) * 1000)[:5], flush=True)
        self.statistics["conflict_time"] = str((end_time - start_time) * 1000)[:4]
        start_time = time.time()
        x = torch.from_numpy(x.astype(np.float32))
        txns, mask = self.model.obtain_schedule(x)
        for txn_idx, ts in txns:
            if txn_idx < len(txns_N):
                if ts >= 0:
                    txn = txns_N[txn_idx]
                    txns_to_schedule.append(txn)
                    if hasattr(self.scheduler, 'memory'):
                        self.scheduler.memory[txn.txn] = self.step + ts # for if using LumpScheduler
                else:
                    self.statistics['n_aborts'] += 1
                    self.txnPool.extend(txns_N[txn_idx])
                # self.scheduler.memory[txn.txn] = self.step + ts

        txns_to_reschedule = np.where(mask == 0)[0].tolist()
        # print(f'scheduling {len(txns_to_schedule)}')
        # print(self.step, self.scheduler.memory, flush=True)
        self.txnPool.extend([txns_N[idx] for idx in txns_to_reschedule])
        self.txnPool = txns_to_schedule + self.txnPool
        
        # self.scheduler.memory = memory
        # print('done updating memory RL', flush=True)
        end_time = time.time()
        self.statistics["RL_time"] = str((end_time - start_time) * 1000)[:4]
        return txns_to_schedule

    
    def sim(self, freeze:bool=False, retryOnAbort:bool=False, n:int=None, T:int=None, ML_RL:str = None) -> dict:
        # print(len(self.scheduled_txn), flush=True)
        # print(len(self.scheduled_txn), len(self.txnPool), len(self.flushPool), flush=True)
        start_time = time.time()
        if self.done() and self.statistics["n_successes"] == 500:
            self.finished = True
            return self.statistics
        
        while len(self.scheduled_txn) > 0 and self.scheduled_txn[0].priority <= self.step:
            if self.scheduled_txn[0].priority < self.step: assert False, "transaction should be scheduled earlier"
            p = heapq.heappop(self.scheduled_txn)
            if hasattr(self.scheduler, 'memory') and self.scheduler.memory is not None:
                del self.scheduler.memory[p.txn.txn]
            self.inflight[p.txn.txn] = p.txn.operations
            self.memo[p.txn.txn] = clone_transaction(p.txn) # clone just in case need to reschedule

        txns_to_schedule = self.txnPool
        if len(self.txnPool) == 0: # no more transactions to be scheduled
            self.tick(retryOnAbort=retryOnAbort)
            return self.statistics

        if ML_RL is not None: #ML model scheduling
            if self.skipT and T is not None and (self.step % T) != 0: #only schedule every T steps
                self.tick(retryOnAbort=retryOnAbort)
                return self.statistics
            if ML_RL == "ML":
                txns_to_schedule = self.update_memory_ML(n, T) # not fully implemented
            elif ML_RL == "RL": 
                txns_to_schedule = self.update_memory_RL(n, T)
        
        # print(txns_to_schedule, flush=True)
        decisions = self.scheduler.schedule(self.inflight, self, txns_to_schedule, self.step) # pass self for flush()
        # print(decisions, flush=True)
        if ML_RL is not None:
            if isinstance(self.scheduler, LumpScheduler):
                decisions.extend([0]*(len(self.txnPool) - len(decisions)))

        end_time = time.time()
        self.statistics["decision_time"] = str((end_time - start_time) * 1000)[:4]

        # assert len(decisions) == len(self.txnPool), f"Decision length is not equal transaction pool length, {type(self.scheduler)}, {len(decisions)}, {len(self.txnPool)}"

        new_pool = []
        # print(decisions, flush=True)

        if isinstance(self.scheduler, KSMFTwistedScheduler) or isinstance(self.scheduler, KSMFTwistedOracle2PhaseDontCareScheduler):
            counter = 0
            for (i, v) in decisions:
                # t = self.txnPool[i]
                t = txns_to_schedule[i]
                if v >= 1: 
                    self.scheduled_time[t.txn] = self.step + v - 1
                    counter += 1
                    if v == 1:
                        self.inflight[t.txn] = t.operations
                        self.memo[t.txn] = clone_transaction(t)
                    else:
                        heapq.heappush(self.scheduled_txn, Pair(self.scheduled_time[t.txn], t))
                elif v == -1:
                    pass # scheduler says toss this transaction away
                elif v == 0: new_pool.append(t)
                else: assert False, f"unknown decision {v}"
            # print(counter, flush=True)
        elif isinstance(self.scheduler, RLSMFTwistedScheduler):
            step_stop = self.step + T
            while self.step < step_stop:
                for (i, v) in decisions:
                    # t = self.txnPool[i]
                    t = txns_to_schedule[i]
                    if v >= 1: 
                        self.scheduled_time[t.txn] = self.step + v - 1
                        if v == 1:
                            self.inflight[t.txn] = t.operations
                            self.memo[t.txn] = clone_transaction(t)
                        else:
                            heapq.heappush(self.scheduled_txn, Pair(self.scheduled_time[t.txn], t))
                    elif v == -1:
                        pass # scheduler says toss this transaction away
                    elif v == 0: new_pool.append(t)
                    else: assert False, f"unknown decision {v}"
                self.txnPool = self.txnPool[len(decisions):] + new_pool + self.flushPool
                self.flushPool = []
                statistics = dict()

                if freeze: statistics |= self.sim_frozen()
                self.tick(retryOnAbort=retryOnAbort) # tick one step
                return (self.statistics | statistics) if freeze else self.statistics
        else:
            for i, t in enumerate(self.txnPool):
            # for i, t in enumerate(txns_to_schedule):
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
                    # new_txn.txn += DELTA
                    del self.memo[txn]
                    if hasattr(self.scheduler, 'memory') and self.scheduler.memory is not None and txn in self.scheduler.memory:
                        del self.scheduler.memory[txn]
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
                    if hasattr(self.scheduler, 'memory') and self.scheduler.memory is not None and txn in self.scheduler.memory:
                        del self.scheduler.memory[txn]

            if len(self.inflight[txn]) == 0:
                del self.inflight[txn]

        self.statistics["steps"] += 1
        self.step += 1
        if isinstance(self.scheduler, LumpScheduler): self.scheduler.step += 1

    def done(self) -> bool:
        return len(self.inflight) == 0 and len(self.txnPool) == 0 and len(self.scheduled_txn) == 0
    
    def print_statistics(self) -> dict:
        print(self.statistics)
        return self.statistics

    def reset_ts_stats(self):
        if self.statistics.get("decision_time"): del self.statistics["decision_time"]
        if self.statistics.get("conflict_time"): del self.statistics["conflict_time"]
        if self.statistics.get("RL_time"): del self.statistics["RL_time"]

    def online_stats(self):
        return self.step, len(self.inflight) + len(self.txnPool) + len(self.scheduled_txn)

