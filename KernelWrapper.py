from utils import conflict

class KernelWrapper():
    def __init__(self, kernel):
        self.kernel = kernel
    
    def __call__(self, txn_list:list, locker, curstep:int) -> list: 
        import numpy as np
        
        num_txn = len(txn_list)
        not_scheduled = []
        kernel_idx = []
        cutoff = num_txn # default to no remnants

        for w in range(num_txn):
            txn = txn_list[w]
            l = len(txn.operations)
            can = True # initially, assume can dispatch transaction now
            for i, op in enumerate(txn.operations):
                ex = locker.probe(op.resource, op.type) + 1 # time when operator executes
                dispatch = ex - i # when the initial operator should be
                if dispatch > curstep: 
                    can = False # can only schedule this transaction sometime in the future, don't care about it now
                    break

            if can: # can schedule
                kernel_idx.append(w)
                if len(kernel_idx) == self.kernel.N: # max that the kernel can handle
                    cutoff = w + 1
                    break
            else:
                not_scheduled.append(txn)
        
        # compute C
        l = len(kernel_idx)
        if l == 0: # nothing can be scheduled
            return [], txn_list
        C = np.zeros((l, l))
        for i in range(l):
            for j in range(l):
                C[i][j] = conflict(txn_list[kernel_idx[i]], txn_list[kernel_idx[j]], 0)
        
        _, res = self.kernel.run(C, debug=False)
        kernel_res = [(txn_list[x].txn, y) for x, y in zip(kernel_idx, res)]
        
        remnant = not_scheduled + [txn_list[kernel_idx[i]] for i in range(l) if kernel_res[i][1]==0] + txn_list[cutoff:]

        return kernel_res, remnant
