from sortedcontainers import SortedSet

class Locker():

    ### A simplified structure to detect conflicts
    #   Optimisation: stores the timestep in which a transaction will release a resource

    def __init__(self, ref=None):
        self.probe_table = dict() # maps resource to a dictionary {"R": t1, "W", t2} indicating the latest time it will be held; used by some schedulers
        if ref is None:
            self.master_table = dict()
            # master_table: resource -> dict(txn -> R/W)
            # the value dictionary is either one 'W' or a list of 'R'
        else:
            # deepclone
            self.master_table = dict()
            for resource in ref.master_table:
                self.master_table[resource] = dict()
                for txn in ref.master_table[resource]:
                    self.master_table[resource][txn] = ref[resource][txn]
            for resource in ref.probe_table:
                self.probe_table[resource] = dict()
                for type in ref.probe_table[resource]:
                    self.probe_table[resource][type] = ref.probe_table[resource][type]

    def probe(self, resource, typ:str) -> int:
        # returns the latest timestep in which a resource is used
        if resource not in self.probe_table: return -1 # no one using it anyways
        if typ == "R": return self.probe_table[resource].get("W", -1) # must be after the last write
        elif typ == "W": return max(self.probe_table[resource].get("R", -1), self.probe_table[resource].get("W", -1)) # must be after last operation
        assert False, "unreachable code"

    def update(self, resource, typ:str, tt:int) -> None:
        if resource not in self.probe_table: self.probe_table[resource] = dict()
        self.probe_table[resource][typ] = max(self.probe_table[resource].get(typ, tt), tt)

    def lock(self, resource, txn:int, typ:str, delta:int) -> bool:
        # delta is the absolute timestep in which this resource must be released by txn

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
            if len(self.master_table[resource]) == 0:
                self.master_table[resource][txn] = "W"
                return True
            if txn in self.master_table[resource] and len(self.master_table[resource]) == 1:
                self.master_table[resource][txn] = "W" # upgrade to most restrictive, might be previously reading
                return True
            return False # no other cases
        else:
            assert False, "invalid transaction type" # not possible

    def remove(self, resource, txn):
        # used when a transaction finished successfully
        if resource in self.master_table and txn in self.master_table[resource]:
            del self.master_table[resource][txn]
            if len(self.master_table[resource]) == 0:
                del self.master_table[resource]
        else:
            assert False, "not possible to remove nonexistent resource"

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

class AdvancedLocker():

    ### A structure to detect conflicts
    #   Optimisation: stores the timestep in which a transaction will release a resource

    def __init__(self, ref=None):
        self.probe_table = dict() 
        # maps resource to a dictionary {"R": s1, "W", s2} 
        # each storing a SortedSet of the time interval
        if ref is None:
            self.master_table = dict()
            # master_table: resource -> dict(txn -> R/W)
            # the value dictionary is either one 'W' or a list of 'R'
        else:
            # deepclone
            self.master_table = dict()
            for resource in ref.master_table:
                self.master_table[resource] = dict()
                for txn in ref.master_table[resource]:
                    self.master_table[resource][txn] = ref[resource][txn]
            for resource in ref.probe_table:
                self.probe_table[resource] = dict()
                for type in ref.probe_table[resource]:
                    self.probe_table[resource][type] = SortedSet()
                    for x in ref.probe_table[resource][type]:
                        self.probe_table[resource][type].add(x)

    def search(self, os, t, period): # want a [t, t+period] inclusive
        if os is None or len(os) == 0: return t
        idx = os.bisect_right((t, -1))
        if idx - 1 >= 0:
            _, val_e = os[idx - 1]
            if val_e >= t: return self.search(os, val_e + 1, period)
        if idx == len(os): return t # t to the right of everything
        val_s, val_e = os[idx]
        if t + period < val_s: return t # can fit into the previous period
        return self.search(os, val_e + 1, period) # must look after this period

    def dual_search(self, os1, os2, t, period):
        t_hat = self.search(os1, t, period) # search one sorted dict first for an estimate
        t_hat_hat = self.search(os2, t_hat, period) # then search the other for another estimate
        assert t_hat >= t and t_hat_hat >= t_hat, f"invariant check: t={t}, t_hat={t_hat}, t_hat_hat={t_hat_hat}"
        if t_hat_hat == t: return t # if both agrees, then the returned time step is possible
        return self.dual_search(os1, os2, t_hat_hat, period) # try again with new upper bound

    def probe(self, resource, typ:str, t:int, period:int) -> int:
        # probes for a time interval [t, t+period], returns the latest time after t that the resource can be used
        if resource not in self.probe_table: return t # no one using it anyways
        if typ == "R": return self.search(self.probe_table[resource].get("W", None), t, period)
        elif typ == "W": return self.dual_search(self.probe_table[resource].get("W", None), self.probe_table[resource].get("R", None), t, period)
        assert False, "unreachable code"

    def update(self, resource, typ:str, t:int, period:int) -> None:
        if resource not in self.probe_table: self.probe_table[resource] = dict()
        if typ not in self.probe_table[resource]: 
            self.probe_table[resource][typ] = SortedSet()
            self.probe_table[resource][typ].add((t, t+period))
            return
        l = len(self.probe_table[resource][typ])
        idx = self.probe_table[resource][typ].bisect_right((t, -1))
        if idx < l:
            a, b = self.probe_table[resource][typ][idx]
            if a <= t + period + 1:
                self.probe_table[resource][typ].remove((a, b))
                period = max(t + period, b) - t
                self.update(resource, typ, t, period)
                return
        if idx - 1 >= 0:
            a, b = self.probe_table[resource][typ][idx - 1]
            if b >= t - 1:
                self.probe_table[resource][typ].remove((a, b))
                period = max(t + period, b) - a
                self.update(resource, typ, a, period)
                return
        self.probe_table[resource][typ].add((t, t+period))

    def lock(self, resource, txn:int, typ:str, delta:int) -> bool:
        # delta is the absolute timestep in which this resource must be released by txn

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
            if len(self.master_table[resource]) == 0:
                self.master_table[resource][txn] = "W"
                return True
            if txn in self.master_table[resource] and len(self.master_table[resource]) == 1:
                self.master_table[resource][txn] = "W" # upgrade to most restrictive, might be previously reading
                return True
            return False # no other cases
        else:
            assert False, "invalid transaction type" # not possible

    def remove(self, resource, txn):
        # used when a transaction finished successfully
        if resource in self.master_table and txn in self.master_table[resource]:
            del self.master_table[resource][txn]
            if len(self.master_table[resource]) == 0:
                del self.master_table[resource]
        else:
            assert False, "not possible to remove nonexistent resource"

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