from abc import ABC, abstractmethod
from utils import Transaction, conflict
from Locker import Locker, AdvancedLocker
import random
from utils import clone_transaction

class Scheduler(ABC):

    @abstractmethod
    def schedule(self, inflight:dict(), Locker, TxnPool:list[Transaction], curstep:int) -> list[int]:
        ...

class RapidFireScheduler(Scheduler):
    def schedule(self, inflight:dict(), Locker, TxnPool:list[Transaction], curstep:int) -> list[int]:
        total_txns = len(TxnPool)
        assert total_txns > 0, "scheduler fail: total number of transactions is 0"
        return [1] + [0]*(total_txns - 1)

class RapidFireRandomScheduler(Scheduler):
    def schedule(self, inflight:dict(), Locker, TxnPool:list[Transaction], curstep:int) -> list[int]:
        total_txns = len(TxnPool)
        assert total_txns > 0, "scheduler fail: total number of transactions is 0"
        res = [0]*total_txns
        res[random.randrange(total_txns)] = 1
        return res

class RapidFireKRandomScheduler(Scheduler):
    def __init__(self, k):
        self.k = k
    def schedule(self, inflight:dict(), Locker, TxnPool:list[Transaction], curstep:int) -> list[int]:
        total_txns = len(TxnPool)
        assert total_txns > 0, "scheduler fail: total number of transactions is 0"
        res = [0]*total_txns
        for _ in range(self.k): res[random.randrange(total_txns)] = 1
        return res

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
                        latest_complete, latest_dispatch = complete, max(curstep, dispatch) # cannot dispatch later than curstep
                latest_complete = max(latest_complete - curstep, 0)
                if shortest_makespan is None or shortest_makespan > latest_complete:
                    shortest_makespan, global_id_smf, local_id_smf, smf_dispatch_time = latest_complete, txnPoolClone[idx][0], idx, latest_dispatch
            res[global_id_smf] = smf_dispatch_time - curstep + 1
            for i, op in enumerate(txnPoolClone[local_id_smf][1].operations):
                self.locker.update(op.resource, op.type, smf_dispatch_time + i)
            txnPoolClone.pop(local_id_smf)
            
        return res

class KSMFTwistedScheduler(Scheduler): # https://www.vldb.org/pvldb/vol17/p2694-cheng.pdf
    def __init__(self, k=5):
        self.k = k # how many transactions to sample
        self.locker = AdvancedLocker()
    
    def schedule(self, inflight:dict(), Locker, TxnPool:list[Transaction], curstep:int) -> list[int]:
        total_txns = len(TxnPool)
        
        assert total_txns > 0, "scheduler fail: total number of transactions is 0"
        res = []

        txnPoolClone = list(enumerate(TxnPool))

        while len(txnPoolClone) >= 1:

            idx_list = random.sample(range(len(txnPoolClone)), min(self.k, len(txnPoolClone)))
            shortest_makespan, global_id_smf, local_id_smf, smf_dispatch_time = None, None, None, None
            for idx in idx_list:
                txn = txnPoolClone[idx][1]
                l = len(txn.operations)
                latest_complete, latest_dispatch = None, None
                t_try = curstep # probing this step
                t_hat = None    # the resultant step
                while t_hat is None or t_hat != t_try:
                    t_hat = t_try
                    for i, op in enumerate(txn.operations):
                        ex = self.locker.probe(op.resource, op.type, t_try + i, op.delta_last) # time when operator executes
                        assert ex >= t_try + i, f"invariance broken: ex={ex}, t_try+i={t_try+i}"
                        t_try = ex - i # when the initial operator should be
                assert t_hat == t_try, f"t_hat={t_hat}, t_try={t_try}"
                assert t_hat >= curstep, f"t_hat={t_hat}"
                dispatch = t_hat
                complete = t_hat + l - 1 # when the entire operation is done
                latest_dispatch = dispatch
                latest_complete = max(complete - curstep, 0)
                if shortest_makespan is None or shortest_makespan > latest_complete:
                    shortest_makespan, global_id_smf, local_id_smf, smf_dispatch_time = latest_complete, txnPoolClone[idx][0], idx, latest_dispatch
            res.append((global_id_smf, smf_dispatch_time - curstep + 1))
            visited = set()
            for i, op in enumerate(txnPoolClone[local_id_smf][1].operations):
                if (op.resource, op.type) in visited: continue
                visited.add((op.resource, op.type))
                self.locker.update(op.resource, op.type, smf_dispatch_time + i, op.delta_last)
            txnPoolClone.pop(local_id_smf)
        assert len(res) == total_txns, "invariance broken"
        return res

class KSMFZeroScheduler(Scheduler): # https://www.vldb.org/pvldb/vol17/p2694-cheng.pdf
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
                        latest_complete, latest_dispatch = complete, max(curstep, dispatch) # cannot dispatch later than curstep
                latest_complete = max(latest_complete - curstep, 0)
                if shortest_makespan is None or shortest_makespan > latest_complete:
                    shortest_makespan, global_id_smf, local_id_smf, smf_dispatch_time = latest_complete, txnPoolClone[idx][0], idx, latest_dispatch
            res[global_id_smf] = smf_dispatch_time - curstep + 1
            if res[global_id_smf] == 1:
                for i, op in enumerate(txnPoolClone[local_id_smf][1].operations):
                    self.locker.update(op.resource, op.type, smf_dispatch_time + i)
            else:
                res[global_id_smf] = 0
            txnPoolClone.pop(local_id_smf)
            
        return res

class KSMFOracle2PhaseDontCareScheduler(Scheduler):
    # Same as KSMF, but knows when correlation are happening and which resource
    # Hypothesis is that this allows for better stacking

    def __init__(self, k=5):
        self.k = k # how many transactions to sample
        self.locker = Locker()
        self.corr_locker = Locker()
        self.corr_resource_list = None
        self.prev = False

    def inject_oracle_list(self, resource_list):
        self.corr_resource_list = resource_list

    def is_corr_resource(self, resource):
        return (self.corr_resource_list is not None) and (resource in self.corr_resource_list)
    
    def schedule(self, inflight:dict(), _, TxnPool:list[Transaction], curstep:int) -> list[int]:
        total_txns = len(TxnPool)
        update_table = dict() # maps resource to how many times it was seen in this iteration
        
        assert total_txns > 0, "scheduler fail: total number of transactions is 0"
        res = [0]*total_txns

        txnPoolClone = list(enumerate(TxnPool))

        firstPhase = []
        secondPhase = []
        for idx, txn in txnPoolClone:
            isCorr = any([self.is_corr_resource(op.resource) for op in txn.operations])
            if isCorr: firstPhase.append((idx, txn))
            else: secondPhase.append((idx, txn))

        firstPhaseLenMemo = len(firstPhase)

        while len(firstPhase) >= 1:
            if self.prev == False: 
                self.locker = Locker(ref=self.corr_locker) # restart, don't care about non-correlated entries if got collision

            self.prev = True

            idx_list = random.sample(range(len(firstPhase)), min(self.k, len(firstPhase)))
            shortest_makespan, global_id_smf, local_id_smf, smf_dispatch_time = None, None, None, None

            for idx in idx_list:
                txn = firstPhase[idx][1]
                l = len(txn.operations)
                latest_complete, latest_dispatch, determined_by_oracle = None, None, False

                for i, op in enumerate(txn.operations):
                    ex = self.locker.probe(op.resource, op.type) + 1 # time when operator executes
                    dispatch = ex - i # when the initial operator should be
                    complete = dispatch + l - 1 # when the entire operation is done
                    update_table[op.resource] = update_table.get(op.resource, 0) + 1 # update
                    if latest_complete is None or complete > latest_complete:
                        latest_complete, latest_dispatch = complete, max(curstep, dispatch) # cannot dispatch later than curstep
                latest_complete = max(latest_complete - curstep, 0)
                if shortest_makespan is None or shortest_makespan > latest_complete:
                    shortest_makespan, global_id_smf, local_id_smf, smf_dispatch_time = latest_complete, firstPhase[idx][0], idx, latest_dispatch
            res[global_id_smf] = smf_dispatch_time - curstep + 1
            for i, op in enumerate(firstPhase[local_id_smf][1].operations):
                self.locker.update(op.resource, op.type, smf_dispatch_time + i)
                self.corr_locker.update(op.resource, op.type, smf_dispatch_time + i)
            firstPhase.pop(local_id_smf)

        if firstPhaseLenMemo == 0:
            self.prev = False

        while len(secondPhase) >= 1:

            idx_list = random.sample(range(len(secondPhase)), min(self.k, len(secondPhase)))
            shortest_makespan, global_id_smf, local_id_smf, smf_dispatch_time = None, None, None, None

            for idx in idx_list:
                txn = secondPhase[idx][1]
                l = len(txn.operations)
                latest_complete, latest_dispatch, determined_by_oracle = None, None, False

                for i, op in enumerate(txn.operations):
                    ex = self.locker.probe(op.resource, op.type) + 1 # time when operator executes
                    dispatch = ex - i # when the initial operator should be
                    complete = dispatch + l - 1 # when the entire operation is done
                    update_table[op.resource] = update_table.get(op.resource, 0) + 1 # update
                    if latest_complete is None or complete > latest_complete:
                        latest_complete, latest_dispatch = complete, max(curstep, dispatch) # cannot dispatch later than curstep
                latest_complete = max(latest_complete - curstep, 0)
                if shortest_makespan is None or shortest_makespan > latest_complete:
                    shortest_makespan, global_id_smf, local_id_smf, smf_dispatch_time = latest_complete, secondPhase[idx][0], idx, latest_dispatch
            res[global_id_smf] = smf_dispatch_time - curstep + 1
            for i, op in enumerate(secondPhase[local_id_smf][1].operations):
                self.locker.update(op.resource, op.type, smf_dispatch_time + i)
            secondPhase.pop(local_id_smf)
            
        return res

class KSMFTwistedOracle2PhaseDontCareScheduler(Scheduler):
    # Same as KSMF, but knows when correlation are happening and which resource
    # Hypothesis is that this allows for better stacking

    def __init__(self, k=5):
        self.k = k # how many transactions to sample
        self.locker = AdvancedLocker()
        self.corr_locker = AdvancedLocker()
        self.corr_resource_list = None
        self.prev = False

    def inject_oracle_list(self, resource_list):
        self.corr_resource_list = resource_list

    def is_corr_resource(self, resource):
        return (self.corr_resource_list is not None) and (resource in self.corr_resource_list)
    
    def schedule(self, inflight:dict(), _, TxnPool:list[Transaction], curstep:int) -> list[int]:
        total_txns = len(TxnPool)
        print(f"total_txns = {total_txns}")
        update_table = dict() # maps resource to how many times it was seen in this iteration
        
        assert total_txns > 0, "scheduler fail: total number of transactions is 0"
        res = []

        txnPoolClone = list(enumerate(TxnPool))

        firstPhase = []
        secondPhase = []
        for idx, txn in txnPoolClone:
            isCorr = any([self.is_corr_resource(op.resource) for op in txn.operations])
            if isCorr: firstPhase.append((idx, txn))
            else: secondPhase.append((idx, txn))

        firstPhaseLenMemo = len(firstPhase)

        while len(firstPhase) >= 1:
            if self.prev == False: 
                self.locker = AdvancedLocker(ref=self.corr_locker) # restart, don't care about non-correlated entries if got collision

            self.prev = True

            # start Twisted

            idx_list = random.sample(range(len(firstPhase)), min(self.k, len(firstPhase)))
            shortest_makespan, global_id_smf, local_id_smf, smf_dispatch_time = None, None, None, None
            for idx in idx_list:
                txn = firstPhase[idx][1]
                l = len(txn.operations)
                latest_complete, latest_dispatch = None, None
                t_try = curstep # probing this step
                t_hat = None    # the resultant step
                while t_hat is None or t_hat != t_try:
                    t_hat = t_try
                    for i, op in enumerate(txn.operations):
                        ex = self.locker.probe(op.resource, op.type, t_try + i, op.delta_last) # time when operator executes
                        assert ex >= t_try + i, f"invariance broken: ex={ex}, t_try+i={t_try+i}"
                        t_try = ex - i # when the initial operator should be
                assert t_hat == t_try, f"t_hat={t_hat}, t_try={t_try}"
                assert t_hat >= curstep, f"t_hat={t_hat}"
                dispatch = t_hat
                complete = t_hat + l - 1 # when the entire operation is done
                latest_dispatch = dispatch
                latest_complete = max(complete - curstep, 0)
                if shortest_makespan is None or shortest_makespan > latest_complete:
                    shortest_makespan, global_id_smf, local_id_smf, smf_dispatch_time = latest_complete, firstPhase[idx][0], idx, latest_dispatch
            res.append((global_id_smf, smf_dispatch_time - curstep + 1))
            visited = set()
            for i, op in enumerate(firstPhase[local_id_smf][1].operations):
                if (op.resource, op.type) in visited: continue
                visited.add((op.resource, op.type))
                self.locker.update(op.resource, op.type, smf_dispatch_time + i, op.delta_last)
                self.corr_locker.update(op.resource, op.type, smf_dispatch_time + i, op.delta_last) # update the base skeleton locker
            firstPhase.pop(local_id_smf)

            # end Twisted

        if firstPhaseLenMemo == 0:
            self.prev = False

        while len(secondPhase) >= 1:

            # start Twisted

            idx_list = random.sample(range(len(secondPhase)), min(self.k, len(secondPhase)))
            shortest_makespan, global_id_smf, local_id_smf, smf_dispatch_time = None, None, None, None
            for idx in idx_list:
                txn = secondPhase[idx][1]
                l = len(txn.operations)
                latest_complete, latest_dispatch = None, None
                t_try = curstep # probing this step
                t_hat = None    # the resultant step
                while t_hat is None or t_hat != t_try:
                    t_hat = t_try
                    for i, op in enumerate(txn.operations):
                        ex = self.locker.probe(op.resource, op.type, t_try + i, op.delta_last) # time when operator executes
                        assert ex >= t_try + i, f"invariance broken: ex={ex}, t_try+i={t_try+i}"
                        t_try = ex - i # when the initial operator should be
                assert t_hat == t_try, f"t_hat={t_hat}, t_try={t_try}"
                assert t_hat >= curstep, f"t_hat={t_hat}"
                dispatch = t_hat
                complete = t_hat + l - 1 # when the entire operation is done
                latest_dispatch = dispatch
                latest_complete = max(complete - curstep, 0)
                if shortest_makespan is None or shortest_makespan > latest_complete:
                    shortest_makespan, global_id_smf, local_id_smf, smf_dispatch_time = latest_complete, secondPhase[idx][0], idx, latest_dispatch
            res.append((global_id_smf, smf_dispatch_time - curstep + 1))
            visited = set()
            for i, op in enumerate(secondPhase[local_id_smf][1].operations):
                if (op.resource, op.type) in visited: continue
                visited.add((op.resource, op.type))
                self.locker.update(op.resource, op.type, smf_dispatch_time + i, op.delta_last)
            secondPhase.pop(local_id_smf)

            # end Twisted
            
        return res

class KSMFOracle2PhaseScheduler(Scheduler):
    # Same as KSMF, but knows when correlation are happening and which resource
    # Hypothesis is that this allows for better stacking

    def __init__(self, k=5):
        self.k = k # how many transactions to sample
        self.locker = Locker()
        self.corr_resource_list = None

    def inject_oracle_list(self, resource_list):
        self.corr_resource_list = resource_list

    def is_corr_resource(self, resource):
        return (self.corr_resource_list is not None) and (resource in self.corr_resource_list)
    
    def schedule(self, inflight:dict(), Locker, TxnPool:list[Transaction], curstep:int) -> list[int]:
        total_txns = len(TxnPool)
        update_table = dict() # maps resource to how many times it was seen in this iteration
        
        assert total_txns > 0, "scheduler fail: total number of transactions is 0"
        res = [0]*total_txns

        txnPoolClone = list(enumerate(TxnPool))

        firstPhase = []
        secondPhase = []
        for idx, txn in txnPoolClone:
            isCorr = any([self.is_corr_resource(op.resource) for op in txn.operations])
            if isCorr: firstPhase.append((idx, txn))
            else: secondPhase.append((idx, txn))

        firstPhase_smf = None
        firstPhaseLenMemo = len(firstPhase)

        while len(firstPhase) >= 1:

            idx_list = random.sample(range(len(firstPhase)), min(self.k, len(firstPhase)))
            shortest_makespan, global_id_smf, local_id_smf, smf_dispatch_time = None, None, None, None

            for idx in idx_list:
                txn = firstPhase[idx][1]
                l = len(txn.operations)
                latest_complete, latest_dispatch, determined_by_oracle = None, None, False

                for i, op in enumerate(txn.operations):
                    ex = self.locker.probe(op.resource, op.type) + 1 # time when operator executes
                    dispatch = ex - i # when the initial operator should be
                    complete = dispatch + l - 1 # when the entire operation is done
                    update_table[op.resource] = update_table.get(op.resource, 0) + 1 # update
                    if latest_complete is None or complete > latest_complete:
                        latest_complete, latest_dispatch = complete, max(curstep, dispatch) # cannot dispatch later than curstep
                latest_complete = max(latest_complete - curstep, 0)
                if shortest_makespan is None or shortest_makespan > latest_complete:
                    shortest_makespan, global_id_smf, local_id_smf, smf_dispatch_time = latest_complete, firstPhase[idx][0], idx, latest_dispatch
            firstPhase_smf = shortest_makespan
            res[global_id_smf] = smf_dispatch_time - curstep + 1
            for i, op in enumerate(firstPhase[local_id_smf][1].operations):
                self.locker.update(op.resource, op.type, smf_dispatch_time + i)
            firstPhase.pop(local_id_smf)

        if firstPhaseLenMemo == 0:

            while len(secondPhase) >= 1:

                idx_list = random.sample(range(len(secondPhase)), min(self.k, len(secondPhase)))
                shortest_makespan, global_id_smf, local_id_smf, smf_dispatch_time = None, None, None, None

                for idx in idx_list:
                    txn = secondPhase[idx][1]
                    l = len(txn.operations)
                    latest_complete, latest_dispatch, determined_by_oracle = None, None, False

                    for i, op in enumerate(txn.operations):
                        ex = self.locker.probe(op.resource, op.type) + 1 # time when operator executes
                        dispatch = ex - i # when the initial operator should be
                        complete = dispatch + l - 1 # when the entire operation is done
                        update_table[op.resource] = update_table.get(op.resource, 0) + 1 # update
                        if latest_complete is None or complete > latest_complete:
                            latest_complete, latest_dispatch = complete, max(curstep, dispatch) # cannot dispatch later than curstep
                    latest_complete = max(latest_complete - curstep, 0)
                    if shortest_makespan is None or shortest_makespan > latest_complete:
                        shortest_makespan, global_id_smf, local_id_smf, smf_dispatch_time = latest_complete, secondPhase[idx][0], idx, latest_dispatch
                res[global_id_smf] = smf_dispatch_time - curstep + 1
                for i, op in enumerate(secondPhase[local_id_smf][1].operations):
                    self.locker.update(op.resource, op.type, smf_dispatch_time + i)
                secondPhase.pop(local_id_smf)
        else:
            while len(secondPhase) >= 1:

                idx_list = random.sample(range(len(secondPhase)), min(self.k, len(secondPhase)))
                shortest_makespan, global_id_smf, local_id_smf, smf_dispatch_time = None, None, None, None

                for idx in idx_list:
                    txn = secondPhase[idx][1]
                    l = len(txn.operations)
                    latest_complete, latest_dispatch, determined_by_oracle = None, None, False

                    for i, op in enumerate(txn.operations):
                        ex = self.locker.probe(op.resource, op.type) + 1 # time when operator executes
                        dispatch = ex - i # when the initial operator should be
                        complete = dispatch + l - 1 # when the entire operation is done
                        update_table[op.resource] = update_table.get(op.resource, 0) + 1 # update
                        if latest_complete is None or complete > latest_complete:
                            latest_complete, latest_dispatch = complete, max(curstep, dispatch) # cannot dispatch later than curstep
                    latest_complete = max(latest_complete - curstep, 0)
                    if shortest_makespan is None or shortest_makespan > latest_complete:
                        shortest_makespan, global_id_smf, local_id_smf, smf_dispatch_time = latest_complete, secondPhase[idx][0], idx, latest_dispatch
                if shortest_makespan <= firstPhase_smf:
                    res[global_id_smf] = smf_dispatch_time - curstep + 1
                    for i, op in enumerate(secondPhase[local_id_smf][1].operations):
                        self.locker.update(op.resource, op.type, smf_dispatch_time + i)
                    secondPhase.pop(local_id_smf)
                else:
                    res[global_id_smf] = 0 # don't schedule yet
            
        return res

class KSMFOracleScheduler(Scheduler):
    # Same as KSMF, but knows when correlation are happening and which resource
    # Hypothesis is that this allows for better stacking

    def __init__(self, k=5):
        self.k = k # how many transactions to sample
        self.locker = Locker()
        self.corr_resource_list = None

    def inject_oracle_list(self, resource_list):
        self.corr_resource_list = resource_list

    def is_corr_resource(self, resource):
        return (self.corr_resource_list is not None) and (resource in self.corr_resource_list)
    
    def schedule(self, inflight:dict(), Locker, TxnPool:list[Transaction], curstep:int) -> list[int]:
        total_txns = len(TxnPool)
        update_table = dict() # maps resource to how many times it was seen in this iteration
        
        assert total_txns > 0, "scheduler fail: total number of transactions is 0"
        res = [0]*total_txns

        txnPoolClone = list(enumerate(TxnPool))

        while len(txnPoolClone) >= 1:

            idx_list = random.sample(range(len(txnPoolClone)), min(self.k, len(txnPoolClone)))
            shortest_makespan, global_id_smf, local_id_smf, smf_dispatch_time = None, None, None, None
            oracle_shortest_makespan, oracle_global_id_smf, oracle_local_id_smf, oracle_smf_dispatch_time = None, None, None, None
            # oracle_override = False # final is determined by oracle

            for idx in idx_list:
                txn = txnPoolClone[idx][1]
                l = len(txn.operations)
                latest_complete, latest_dispatch, determined_by_oracle = None, None, False

                for i, op in enumerate(txn.operations):
                    ex = self.locker.probe(op.resource, op.type) + 1 # time when operator executes
                    dispatch = ex - i # when the initial operator should be
                    complete = dispatch + l - 1 # when the entire operation is done
                    update_table[op.resource] = update_table.get(op.resource, 0) + 1 # update
                    if latest_complete is None or complete > latest_complete:
                        determined_by_oracle = self.is_corr_resource(op.resource)
                        latest_complete, latest_dispatch = complete, max(curstep, dispatch) # cannot dispatch later than curstep
                latest_complete = max(latest_complete - curstep, 0)
                if shortest_makespan is None or shortest_makespan > latest_complete:
                    shortest_makespan, global_id_smf, local_id_smf, smf_dispatch_time = latest_complete, txnPoolClone[idx][0], idx, latest_dispatch
                if not determined_by_oracle:
                    if oracle_shortest_makespan is None or oracle_shortest_makespan > latest_complete:
                        oracle_shortest_makespan, oracle_global_id_smf, oracle_local_id_smf, oracle_smf_dispatch_time = latest_complete, txnPoolClone[idx][0], idx, latest_dispatch
                # if determined_by_oracle:
                #     if oracle_override:
                #         # if already exists a transaction having oracle override, then SMF
                #         if shortest_makespan is None or shortest_makespan > latest_complete:
                #             shortest_makespan, global_id_smf, local_id_smf, smf_dispatch_time = latest_complete, txnPoolClone[idx][0], idx, latest_dispatch
                #     else:
                #         # else, enable oracle override
                #         oracle_override = True
                #         shortest_makespan, global_id_smf, local_id_smf, smf_dispatch_time = latest_complete, txnPoolClone[idx][0], idx, latest_dispatch
                # elif (not oracle_override) and (shortest_makespan is None or shortest_makespan > latest_complete):
                #     shortest_makespan, global_id_smf, local_id_smf, smf_dispatch_time = latest_complete, txnPoolClone[idx][0], idx, latest_dispatch
            if oracle_shortest_makespan is not None:
                if random.random() < 0.1: # HYPERPARAMETER
                    shortest_makespan, global_id_smf, local_id_smf, smf_dispatch_time = oracle_shortest_makespan, oracle_global_id_smf, oracle_local_id_smf, oracle_smf_dispatch_time
            res[global_id_smf] = smf_dispatch_time - curstep + 1
            for i, op in enumerate(txnPoolClone[local_id_smf][1].operations):
                self.locker.update(op.resource, op.type, smf_dispatch_time + i)
            txnPoolClone.pop(local_id_smf)
            
        return res

class QueueBasedScheduler(Scheduler):

    def __init__(self, n_queues:int):
        self.n_queues = n_queues
        self.queues = [[] for _ in range(n_queues)] # a queue of transaction ids
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

class QueueScheduler(Scheduler):
    def __init__(self, n_queues:int, scheduler_class, **kwargs):
        self.n_queues = n_queues
        self.queues:list[list[Transaction]] = [[] for _ in range(n_queues)] # a queue of transaction ids
        self.schedulers = [scheduler_class(**kwargs) for _ in range(n_queues)] # one scheduler for each queue
        self.seen = set()
        self.locker = Locker()

    def assign_to_queue(self, txn:Transaction):
        queue_num = txn.operations[0].resource[1] % self.n_queues # currently, just hash the row id of the first operation
        self.queues[queue_num].append(clone_transaction(txn))

    def schedule(self, inflight:dict(), Locker, TxnPool:list[Transaction], curstep:int) -> list[int]:
        total_txns = len(TxnPool)
        res = [0]*total_txns
        mapper = dict() # maps transaction ID to index in the queue, so that return array is matched up

        # 1. Schedule new transactions
        for i, txn in enumerate(TxnPool):
            if txn.txn not in self.seen:
                self.assign_to_queue(txn)
                self.seen.add(txn.txn)
            mapper[txn.txn] = i
        #print(mapper)
        #print([len(q) for q in self.queues])

        # 2. Process kernels
        for i in range(self.n_queues):
            if len(self.queues[i]) == 0: continue
            cur_res = self.schedulers[i].schedule(None, None, self.queues[i], curstep)
            for j, txn in enumerate(self.queues[i]):
                if txn.txn not in mapper: continue # already in queue, not scheduled
                res[mapper[txn.txn]] = cur_res[j]
            
        return res

class QueueKernelScheduler(Scheduler):
    def __init__(self, n_queues:int, kernel):
        self.n_queues = n_queues
        self.queues:list[list[Transaction]] = [[] for _ in range(n_queues)] # a queue of transaction ids
        self.kernel = kernel
        self.seen = set()
        self.locker = Locker()

    def assign_to_queue(self, txn:Transaction):
        queue_num = txn.operations[0].resource[1] % self.n_queues # currently, just hash the row id of the first operation
        self.queues[queue_num].append(clone_transaction(txn))

    def schedule(self, inflight:dict(), Locker, TxnPool:list[Transaction], curstep:int) -> list[int]:
        total_txns = len(TxnPool)
        res = [0]*total_txns
        mapper = dict() # maps transaction ID to index in the queue, so that return array is matched up

        # 1. Schedule new transactions
        for i, txn in enumerate(TxnPool):
            if txn.txn not in self.seen:
                self.assign_to_queue(txn)
                self.seen.add(txn.txn)
            mapper[txn.txn] = i
        #print(mapper)
        #print([len(q) for q in self.queues])

        # 2. Process kernels
        for i in range(self.n_queues):
            if len(self.queues[i]) == 0: continue
            kernel_res, remnant = self.kernel(self.queues[i], self.locker, curstep) # must pass in locker to filter those that can't be scheduled at the current time step
            # how many did the kernel look at, result for those looked at, remaining unprocessed
            self.queues[i] = remnant # update queue
            for x, t in kernel_res: 
                res[mapper[x]] = t
            
        return res

class IntegerOptScheduler(Scheduler):
    # at each time step, schedule as much as possible, the rest are deferred to the next
    # maintains a state of resources used in Locker. If cannot schedule, then don't
    pass

class LumpScheduler(Scheduler):
    def __init__(self):
        self.memory:dict = {}
        self.step = 0
        self.dont_care = None
    
    def inject_memory(self, memory:dict, dont_care:int):
        self.memory = memory
        self.dont_care = dont_care
    
    def schedule(self, inflight:dict(), Locker, TxnPool:list[Transaction], curstep:int) -> list[int]:
        total_txns = len(TxnPool)
        assert total_txns > 0, "scheduler fail: total number of transactions is 0"
        res = [0]*total_txns
        #print(self.memory, flush=True)
        for i, txn in enumerate(TxnPool):
            if txn.txn not in self.memory: 
                #assert False, "transaction not scheduled"
                res[i] = 0
            else:
                ts = self.memory[txn.txn]
                if self.step > ts: 
                    # assert False, "transaction should have been scheduled earlier"
                    res[i] = 0
                if self.dont_care is not None and self.dont_care == ts: 
                    res[i] = -1
                elif self.step == ts: 
                    res[i] = 1 # schedule it
        self.step += 1
        return res


class RLSMFTwistedScheduler(Scheduler): # https://www.vldb.org/pvldb/vol17/p2694-cheng.pdf
    def __init__(self, k=5):
        self.k = k # how many transactions to sample
        self.locker = AdvancedLocker()
    
    def schedule(self, inflight:dict(), Locker, TxnPool:list[Transaction], curstep:int) -> list[int]:
        total_txns = len(TxnPool)
        
        assert total_txns > 0, "scheduler fail: total number of transactions is 0"
        res = []

        # txnPoolClone = list(enumerate(TxnPool))
        # print(txnPoolClone, flush=True)

        # while len(txnPoolClone) >= 1:

        #idx_list = random.sample(range(len(txnPoolClone)), min(self.k, len(txnPoolClone)))
        shortest_makespan, global_id_smf, _, smf_dispatch_time = None, None, None, None
        idx_list = range(len(TxnPool))
        for idx, txn in enumerate(TxnPool):
            # txn = txnPoolClone[idx][1]
            #print(idx, txn, flush=True)
            #print(txnPoolClone[idx][0], flush=True)
            #txn = txnPoolClone[txn_info][1]
            # print(idx, txn_info, flush=True)
            l = len(txn.operations)
            latest_complete, latest_dispatch = None, None
            t_try = curstep # probing this step
            t_hat = None    # the resultant step
            while t_hat is None or t_hat != t_try:
                t_hat = t_try
                for i, op in enumerate(txn.operations):
                    ex = self.locker.probe(op.resource, op.type, t_try + i, op.delta_last) # time when operator executes
                    assert ex >= t_try + i, f"invariance broken: ex={ex}, t_try+i={t_try+i}"
                    t_try = ex - i # when the initial operator should be
            assert t_hat == t_try, f"t_hat={t_hat}, t_try={t_try}"
            assert t_hat >= curstep, f"t_hat={t_hat}"
            dispatch = t_hat
            complete = t_hat + l - 1 # when the entire operation is done
            latest_dispatch = dispatch
            latest_complete = max(complete - curstep, 0)

            res.append((idx, latest_dispatch - curstep + 1))
            visited = set()
            for i, op in enumerate(txn.operations):
                if (op.resource, op.type) in visited: continue
                visited.add((op.resource, op.type))
                self.locker.update(op.resource, op.type, latest_dispatch + i, op.delta_last)


        #     if shortest_makespan is None or shortest_makespan > latest_complete:
        #         shortest_makespan, global_id_smf, _, smf_dispatch_time = latest_complete, idx, None, latest_dispatch
        # res.append((global_id_smf, smf_dispatch_time - curstep + 1))
        # visited = set()
        # # for i, op in enumerate(txnPoolClone[local_id_smf][1].operations):
        # for i, op in enumerate(txn.operations):
        #     if (op.resource, op.type) in visited: continue
        #     visited.add((op.resource, op.type))
        #     self.locker.update(op.resource, op.type, smf_dispatch_time + i, op.delta_last)
        # # txnPoolClone.pop(local_id_smf)
        # print(len(res), total_txns, flush=True)
        # assert len(res) == total_txns, "invariance broken"
        return res