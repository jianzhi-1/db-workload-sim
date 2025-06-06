from abc import ABC, abstractmethod

class Kernel(ABC):
    def __init__(self, N:int, T:int):
        self.N = N
        self.T = T

    @abstractmethod
    def run(self, C) -> list: 
        # C: conflict matrix i.e. C[i, j, T + t] = 1 if transaction i and j will conflict given offset by t in [-T, ..., T]
        # returns: list denoting what time step between 0 to T a transaction should be scheduled at
        ...

class IntegerOptimisationKernelMkII(Kernel):
    def __init__(self, N):
        super().__init__(N, 0)

    def run(self, C, debug=False):
        assert len(C.shape) == 2, "C is just supposed to be a 2D conflict matrix, not considering multiple timesteps"
        assert C.shape[0] == C.shape[1], "conflict matrix must be a square"
        
        import cvxpy as cp

        N = C.shape[0] # convenience

        # 0. Optimisation variables
        x = cp.Variable(N, boolean=True) # x[i] = 1{transaction i scheduled}

        # 1. Constraints set up
        # 1a. Basic constraints
        constraints = [0 <= x, x <= 1] # either schedule or don't schedule
        if debug: print("basic constraints done")

        # 1b. Conflict constraints
        for i in range(N):
            for j in range(i + 1, N):
                if C[i][j] == 1: # if conflict when scheduled together
                    constraints.append(x[i] + x[j] <= 1) # x[i] + x[j] <= 1
        if debug: print("conflict constraints done")

        # 2. Define optimisation problem
        objective = cp.Maximize(cp.sum(x)) # maximise the number of transactions scheduled
        prob = cp.Problem(objective, constraints)

        # 3. Solve
        res = [0]*N # default not scheduled
        throughput = int(prob.solve(solver=cp.GLPK_MI)) # how many transactions were scheduled
        
        for i in range(N):
            if x.value[i] > 0.5:
                res[i] = 1 # scheduled
        return throughput, res

class IntegerOptimisationKernel(Kernel):
    def __init__(self, N, T):
        super().__init__(N, T)

    def run(self, C, debug=False):
        import cvxpy as cp

        C = C.squeeze() # squeeze out the batch dimension

        N, T = self.N, self.T # convenience

        # 0. Optimisation variables
        x = cp.Variable(N*(T + 1), boolean=True) # x[i, t] = transaction i scheduled at time t in [0, ..., T] = x[i*(T + 1) + t]

        # 1. Constraints set up
        # 1a. Basic constraints
        constraints = [0 <= x, x <= 1] # either schedule or don't schedule
        if debug: print("basic constraints done")

        # 1b. Conflict constraints
        for i in range(N):
            for j in range(i + 1, N, 1):
                for t in range(-T, T+1, 1):
                    if C[i][j][T + t] == 1:
                        for tt in range(0, T+1, 1):
                            if tt + t < 0 or tt + t > T: continue
                            constraints.append(x[i*(T+1) + tt] + x[j*(T+1) + tt + t] <= 1) # x[i, tt] + x[j, tt + t] <= 1
        if debug: print("conflict constraints done")

        # 1c. Transaction constraints
        for i in range(N):
            acc = x[i*(T + 1) + 0] # x[i, 0]
            for t in range(1, T+1, 1): acc += x[i*(T + 1) + t] # x[i, t]
            constraints.append(acc <= 1)
        if debug: print("transaction constraints done")

        # 2. Define optimisation problem
        objective = cp.Maximize(cp.sum(x))
        prob = cp.Problem(objective, constraints)

        # 3. Solve
        res = [-1]*N # default not scheduled
        throughput = int(prob.solve(solver=cp.GLPK_MI)) # how many transactions were scheduled
        
        for i in range(N):
            for t in range(T):
                if x.value[i*(T + 1) + t] > 0.5:
                    res[i] = t
                    break
        return throughput, res

class SMFKernel(Kernel): # Cheng, 2024: https://www.vldb.org/pvldb/vol17/p2694-cheng.pdf
    def __init__(self, N, T):
        super().__init__(N, T)

    def run(self, C, debug=False):
        import random
        random.seed(262)
        N, T = self.N, self.T # convenience
        C = C.squeeze() # squeeze out the batch dimension

        order = [i for i in range(N)]
        random.shuffle(order)

        res = [-1]*N
        scheduled = set()

        for i in order:
            # candidate is transaction i
            success, pos = False, -1
            for t in range(0, T+1, 1):
                # try to schedule transaction i at time t
                temp_success = True
                for j, tj in scheduled:
                    if C[i][j][T + (tj - t)] == 1:
                        temp_success = False
                        break
                if temp_success: # greedily assign
                    success, pos = True, t
                    break
            if success:
                res[i] = pos
                scheduled.add((i, pos))

        return len(scheduled), res

class NNKernel(Kernel):
    def __init__(self, N, T, model_file):
        import torch
        from models import LinearModel
        super().__init__(N, T)
        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        self.model = LinearModel(N, T).to(self.device)
        self.model.load_state_dict(torch.load(model_file))
        
    def run(self, C, debug=False):
        import torch
        N, T = self.N, self.T
        C = torch.from_numpy(C).to(self.device)
        C = C.to(torch.float32)
        lamb = self.model(C)
        assert lamb.shape == (1, N, T+2)
        
        res = torch.argmax(lamb, dim=2)
        res[res == T+1] = -1 # not scheduled
        throughput = (res >= 0).sum()
        return throughput, res   
