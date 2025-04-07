from Scheduler import Scheduler
from utils import Transaction, Operation, ReadOperation, WriteOperation

def clone_operation_list(ls:list[Operation]):
    return [ReadOperation(x.txn, x.resource, x.is_last, x.is_last_on_resource) if x.is_read else WriteOperation(x.txn, x.resource, x.is_last, x.is_last_on_resource) for x in ls]

class Locker():

    ### A simplified structure to detect conflicts

    def __init__(self, ref=None):
        if ref is None:
            self.master_table = dict()
            # master_table: resource -> dict(txn -> R/W)
            # the value dictionary is either one 'W' or a list of 'R'
        else:
            # deepclone
            self.master_table = dict()
            for resource in ref:
                self.master_table[resource] = dict()
                for txn in ref[resource]:
                    self.master_table[resource][txn] = ref[resource][txn]

    def lock(self, resource, txn, typ) -> bool:
        if resource not in self.master_table: # if first person, then OK
            self.master_table[resource] = dict()
            self.master_table[resource][txn] = typ
            return True
        
        if typ == "R":
            if txn in self.master_table[resource] and len(self.master_table[resource]) == 1: 
                # i.e. it is the only transaction using the resource anyways
                # self.master_table[resource][txn] = "W" or "R"; either way keep it as it is
                return True

            if txn not in self.master_table[resource]:
                if len(self.master_table[resource]) == 0:
                    self.master_table[resource][txn] = typ
                    return True
                else:
                    val_ls = list(self.master_table[resource].values())
                    if all([elem == "R" for elem in val_ls]):
                        self.master_table[resource][txn] = typ
                        return True
                    return False # someone is writing
                assert False, "unreachable code"

            # else, multiple transactions are using the resource
            val_ls = list(self.master_table[resource].values())
            if all([elem == "R" for elem in val_ls]):
                self.master_table[resource][txn] = typ
                return True

            return False # someone is writing

        elif typ == "W":
            if txn in self.master_table[resource] and len(self.master_table[resource]) == 1:
                self.master_table[resource][txn] = "W" # upgrade to most restrictive, might be previously reading
                return True
            
            return False # no other cases
        else:
            assert(False) # not possible

    def remove(self, resource, txn):
        # used when a transaction finished successfully
        if resource in self.master_table and txn in self.master_table[resource]:
            del self.master_table[resource][txn]
            if len(self.master_table[resource]) == 0:
                del self.master_table[resource]

    def remove_all(self, txn):
        # ideally use sparingly, just search through the resources the transaction used
        res_list = list(self.master_table.keys())
        for resource in res_list:
            if txn in self.master_table[resource]:
                del self.master_table[resource][txn]
                if len(self.master_table[resource]) == 0:
                    del self.master_table[resource]

    def clear(self):
        self.master_table = dict()

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
        self.scheduled_time = dict()
        self.step = 0
        self.clear()

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
        self.txnPool += more_txns

    def sim(self, freeze=False) -> dict:

        if len(self.txnPool) == 0: # no more transactions to be scheduled
            self.tick()
            return self.statistics

        decisions = self.scheduler.schedule(self.inflight, self.resource_locks, self.txnPool)
        assert len(decisions) == len(self.txnPool), "Decision length is not equal transaction pool length"

        new_pool = []
        for i, t in enumerate(self.txnPool):
            if decisions[i] == 1: 
                self.inflight[t.txn] = t.operations
                self.scheduled_time[t.txn] = self.step
            elif decisions[i] == 2:
                pass # schedule says toss this transaction away
            else: new_pool.append(t)
        self.txnPool = new_pool

        statistics = dict()

        if freeze: statistics |= self.sim_frozen()
        self.tick() # tick one step

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
                can = locker_clone.lock(op.resource, op.txn, op.type)
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

    def tick(self) -> dict:

        inflight_keyset = list(self.inflight.keys())
        
        for txn in inflight_keyset:
            if len(self.inflight[txn]) == 0: assert False, "empty transaction for some reason"
            op = self.inflight[txn][0] # get the first operation
            can = self.resource_locks.lock(op.resource, op.txn, op.type)
            if not can: # conflict
                self.result_statistics[txn] = 0
                self.statistics["n_aborts"] += 1
                self.resource_locks.remove_all(op.txn)
                self.inflight[txn] = []
            else: # success
                self.inflight[txn] = self.inflight[txn][1:] # pop the first operations
                if op.is_last_on_resource:
                    self.resource_locks.remove(op.resource, op.txn)

                if op.is_last:
                    self.result_statistics[txn] = 1
                    self.resource_locks.remove_all(op.txn) # not necessary, but for peace of mind
                    self.statistics["n_successes"] += 1

            if len(self.inflight[txn]) == 0:
                del self.inflight[txn]

        self.statistics["steps"] += 1
        self.step += 1

    def done(self) -> bool:
        return len(self.inflight) == 0 and len(self.txnPool) == 0
    
    def print_statistics(self) -> None:
        print(self.statistics)

