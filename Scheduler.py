from abc import ABC, abstractmethod
from utils import Transaction
import random

class Scheduler(ABC):

    @abstractmethod
    def schedule(self, inflight:dict(), Locker, TxnPool:list[Transaction]) -> list[int]:
        ...

class RapidFireScheduler(Scheduler):
    def schedule(self, inflight:dict(), Locker, TxnPool:list[Transaction]) -> list[int]:
        total_txns = len(TxnPool)
        assert total_txns > 0, "scheduler fail: total number of transactions is 0"
        return [1] + [0]*(total_txns - 1)

class SequentialScheduler(Scheduler):
    def schedule(self, inflight:dict(), Locker, TxnPool:list[Transaction]) -> list[int]:
        total_txns = len(TxnPool)
        assert total_txns > 0, "scheduler fail: total number of transactions is 0"
        if len(inflight) > 0: return [0]*total_txns
        return [1] + [0]*(total_txns - 1)

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

    def schedule(self, inflight:dict(), Locker, TxnPool:list[Transaction]) -> list[int]:

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
    
    def schedule(self, inflight:dict(), Locker, TxnPool:list[Transaction]) -> list[int]:
        total_txns = len(TxnPool)
        assert total_txns > 0, "scheduler fail: total number of transactions is 0"
        res = [0]*total_txns
        for i, txn in enumerate(TxnPool):
            if txn.txn not in self.memory: assert False, "transaction not scheduled"
            ts = self.memory[txn.txn]
            if self.step > ts: assert False, "transaction should have been scheduled earlier"
            if self.dont_care is not None and self.dont_care == ts: 
                res[i] = 2
            elif self.step == ts: 
                res[i] = 1 # schedule it
        self.step += 1
        return res
