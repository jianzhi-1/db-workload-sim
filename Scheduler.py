from abc import ABC, abstractmethod
from utils import Transaction, conflict
from Locker import Locker
import random

class Scheduler(ABC):

    @abstractmethod
    def schedule(self, inflight:dict(), Locker, TxnPool:list[Transaction], curstep:int) -> list[int]:
        ...

class RapidFireScheduler(Scheduler):
    def schedule(self, inflight:dict(), Locker, TxnPool:list[Transaction], curstep:int) -> list[int]:
        total_txns = len(TxnPool)
        assert total_txns > 0, "scheduler fail: total number of transactions is 0"
        return [1] + [0]*(total_txns - 1)

class SequentialScheduler(Scheduler):
    def schedule(self, inflight:dict(), Locker, TxnPool:list[Transaction], curstep:int) -> list[int]:
        total_txns = len(TxnPool)
        assert total_txns > 0, "scheduler fail: total number of transactions is 0"
        if len(inflight) > 0: return [0]*total_txns
        return [1] + [0]*(total_txns - 1)

class KSMFScheduler(Scheduler): # https://www.vldb.org/pvldb/vol17/p2694-cheng.pdf
    def __init__(self, k=5):
        self.k = k # how many transactions to sample
        self.locker = Locker()
    
    def schedule(self, inflight:dict(), Locker, TxnPool:list[Transaction], curstep:int) -> list[int]:
        total_txns = len(TxnPool)
        
        assert total_txns > 0, "scheduler fail: total number of transactions is 0"
        res = [0]*total_txns

        txnPoolClone = list(enumerate(TxnPool))

        while len(txnPoolClone) >= 1:

            idx_list = random.sample(range(len(txnPoolClone)), min(self.k, len(txnPoolClone)))
            shortest_makespan, global_id_smf, local_id_smf, smf_dispatch_time = None, None, None, None

            for idx in idx_list:
                txn = txnPoolClone[idx][1]
                l = len(txn.operations)
                latest_complete, latest_dispatch = None, None
                for i, op in enumerate(txn.operations):
                    ex = self.locker.probe(op.resource, op.type) + 1 # time when operator executes
                    dispatch = ex - i # when the initial operator should be
                    complete = dispatch + l - 1 # when the entire operation is done
                    if latest_complete is None or complete > latest_complete:
                        latest_complete, latest_dispatch = complete, dispatch
                latest_complete = max(latest_complete - curstep, 0)
                if shortest_makespan is None or shortest_makespan > latest_complete:
                    shortest_makespan, global_id_smf, local_id_smf, smf_dispatch_time = latest_complete, txnPoolClone[idx][0], idx, latest_dispatch
            res[global_id_smf] = smf_dispatch_time - curstep + 1
            for i, op in enumerate(txnPoolClone[local_id_smf][1].operations):
                self.locker.update(op.resource, op.type, smf_dispatch_time + i)
            txnPoolClone.pop(local_id_smf)
            
        return res

class QueueBasedScheduler(Scheduler):

    def __init__(self, n_queues:int):
        self.n_queues = n_queues
        self.queues = [[]]*n_queues # a queue of transaction ids
        self.started = False
        self.mapper = dict() # maps resource to (queue number, time)
        self.queue_length_mapper = dict()

    def shave(self) -> list[int]:
        res = []
        for i in range(self.n_queues):
            if len(self.queues[i]) == 0: continue
            res.append(self.queues[i][0])
            self.queues[i] = self.queues[i][1:] # pop
        return res

    def schedule(self, inflight:dict(), Locker, TxnPool:list[Transaction], curstep:int) -> list[int]:

        if not self.started:
            for i in range(len(TxnPool)):

                # 1. Find the queue with the latest contention, random if no such queue
                latest_queue, latest_time = None, None
                for op in TxnPool[i].operations:
                    if op.resource in self.mapper:
                        q, t = self.mapper[op.resource]
                        if latest_queue is None or t > latest_time:
                            latest_queue, latest_time = q, t
                if latest_queue is None:
                    latest_queue = random.randint(0, self.n_queues - 1)
                
                # 2. Assign transaction to that queue
                self.queues[latest_queue].append(TxnPool[i].txn)

                # 3. Update the queue and operations meta
                self.queue_length_mapper[latest_queue] = self.queue_length_mapper.get(latest_queue, 0) + len(TxnPool[i].operations)
                set_t = self.queue_length_mapper[latest_queue]

                for op in TxnPool[i].operations:
                    if op.resource not in self.mapper or (op.resource in self.mapper and self.mapper[op.resource][1] < set_t):
                        self.mapper[op.resource] = (latest_queue, set_t)
        
        res = self.shave()
        return [1 if t.txn in res else 0 for t in TxnPool]

class LumpScheduler(Scheduler):
    def __init__(self):
        self.memory:dict = None
        self.step = 0
        self.dont_care = None
    
    def inject_memory(self, memory:dict, dont_care:int):
        self.memory = memory
        self.dont_care = dont_care
    
    def schedule(self, inflight:dict(), Locker, TxnPool:list[Transaction], curstep:int) -> list[int]:
        total_txns = len(TxnPool)
        assert total_txns > 0, "scheduler fail: total number of transactions is 0"
        res = [0]*total_txns
        for i, txn in enumerate(TxnPool):
            if txn.txn not in self.memory: assert False, "transaction not scheduled"
            ts = self.memory[txn.txn]
            if self.step > ts: assert False, "transaction should have been scheduled earlier"
            if self.dont_care is not None and self.dont_care == ts: 
                res[i] = -1
            elif self.step == ts: 
                res[i] = 1 # schedule it
        self.step += 1
        return res


class LumpQueueScheduler(Scheduler):
    def __init__(self, n_queues:int, hotkeys):
        self.memory:dict = None
        self.step = 0
        self.dont_care = None
        self.n_queues = n_queues
        # self.queues = [[]]*n_queues # a queue of transaction ids
        self.queues = {hotkey: [] for hotkey in hotkeys}
        self.started = False
        self.mapper = dict() # maps resource to (queue number, time)
        self.queue_length_mapper = dict()

    def inject_memory(self, memory:dict, dont_care:int):
        self.memory = memory
        self.dont_care = dont_care

    def shave(self) -> list[int]:
        res = []
        for hotkey in self.queues.keys():
            if len(self.queues[hotkey]) == 0: continue
            res.append(self.queues[hotkey][0])
            self.queues[hotkey] = self.queues[hotkey][1:] # pop
        return res

    def schedule(self, inflight:dict(), Locker, TxnPool:list[Transaction]) -> list[int]:
        total_txns = len(TxnPool)
        assert total_txns > 0, "scheduler fail: total number of transactions is 0"
        res_lump = [0]*total_txns
        if not self.started:
            for i in range(len(TxnPool)):

                # 1. Find the queue with the latest contention, random if no such queue
                latest_queue, latest_time = None, None
                for op in TxnPool[i].operations:
                    if op.resource in self.mapper:
                        q, t = self.mapper[op.resource]
                        if latest_queue is None or t > latest_time:
                            latest_queue, latest_time = q, t

                if latest_queue is None and len(TxnPool[i].hotKeys) == 0: # Not a hot key, lump schedule
                    txn = TxnPool[i]
                    if txn.txn not in self.memory: assert False, "transaction not scheduled"
                    ts = self.memory[txn.txn]
                    if self.step > ts: assert False, "transaction should have been scheduled earlier"
                    if self.dont_care is not None and self.dont_care == ts:
                        res_lump[i] = 2
                    elif self.step == ts:
                        res_lump[i] = 1 # schedule it

                else: # hot key, lump schedule
                    if latest_queue is None:
                        latest_queue = TxnPool[i].hotkeys[0]
                    # 2. Assign transaction to that queue
                    self.queues[latest_queue].append(TxnPool[i].txn)

                    # 3. Update the queue and operations meta
                    self.queue_length_mapper[latest_queue] = self.queue_length_mapper.get(latest_queue, 0) + len(TxnPool[i].operations)
                    set_t = self.queue_length_mapper[latest_queue]

                    for op in TxnPool[i].operations:
                        if op.resource not in self.mapper or (op.resource in self.mapper and self.mapper[op.resource][1] < set_t):
                            self.mapper[op.resource] = (latest_queue, set_t)

        res_queue = self.shave()
        self.step += 1
        res_queue = [1 if t.txn in res_queue else 0 for t in TxnPool]
        return res_lump + res_queue
