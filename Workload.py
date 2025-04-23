from abc import ABC, abstractmethod
from utils import Transaction, ReadOperation, WriteOperation
import random
from numpy.random import zipf # reduce importing large numpy library

def draw_zipf_with_max(a, max_value):
    for _ in range(100):
        sample = zipf(a)
        if sample <= max_value: return sample
    return 1

class Workload(ABC):

    @abstractmethod
    def generate_random_transactions(self, num_txns:int, probabilities:list[float], start=0) -> list[Transaction]:
        ...

class SmallBankWorkload(Workload):
    def __init__(self, correlated:bool=False, corr_params:dict=None) -> None:

        self.NUM_ACCOUNTS = 10
        
        self.SMALLBANK_TXN_OPS = { # (Read/Write, Table ID, column_name)
            'Amalgamate': [(False, 0, 'custId0'), (False, 0, 'custId1'), (False, 1, 'custId0'), (False, 2, 'custId1'), (True, 2, 'custId0'), (True, 2, 'custId1')],
            'Balance': [(False, 0, 'custName'), (False, 1, 'custName'), (False, 2, 'custName')],
            'DepositChecking': [(False, 0, 'custName'), (True, 0, 'custName')],
            'SendPayment': [(False, 0, 'sendAcct'), (False, 0, 'destAcct'), (False, 2, 'sendAcct'), (True, 2, 'sendAcct'), (True, 2, 'destAcct')],
            'TransactSavings': [(False, 0, 'custName'), (False, 1, 'custName'), (True, 1, 'custName')],
            'WriteCheck': [(False, 0, 'custName'), (False, 1, 'custName'), (False, 2, 'custName'), (True, 2, 'custName')],
        }

        self.num_txn_types = len(self.SMALLBANK_TXN_OPS)

        self.SMALLBANK_INPUT_TYPINGS = { # (column_name, column_type)
            'Amalgamate': {'custId0': 'long', 'custId1': 'long'},
            'Balance': {'custName': 'string'},
            'DepositChecking': {'custName': 'string', 'amount': 'double'},
            'SendPayment': {'sendAcct': 'long', 'destAcct': 'long', 'amount': 'double'},
            'TransactSavings': {'custName': 'string', 'amount': 'double'},
            'WriteCheck': {'custName': 'string', 'amount': 'double'},
        }

        self.SMALLBANK_HOT_KEYS = [(0,i) for i in range(10)] + [(1, i) for i in range(10)] + [(2, i) for i in range(10)] # ???

        self.correlated:bool = correlated
        self.corr_params:dict = corr_params
        self.ttl = None # time to next event
        self.state = 0 # 0: random, 1: correlated
        self.sticky_value = None # the value that is correlated to during that time interval

    def gen(self, typ:str) -> int:
        if typ in ["long", "string"]: return draw_zipf_with_max(1.01, self.NUM_ACCOUNTS)
        elif typ in "double": return random.randint(0, 100)
        assert False, f"Error: type {typ} not supported in gen"

    def generate_transactions(self, num_txns:int, probabilities:list[float]=None, start=0) -> list[Transaction]:
        if not self.correlated: return self.generate_random_transactions(num_txns, probabilities=probabilities, start=start)
        else:
            import numpy as np
            res = None
            if self.ttl is None or (self.ttl == 0 and self.state == 1) or (self.ttl > 0 and self.state == 0):
                self.state = 0
                if self.ttl is None or self.ttl == 0: self.ttl = max(1, int(np.random.exponential(scale=self.corr_params["normal_scale"], size=None)))
                res = self.generate_random_transactions(num_txns, probabilities=probabilities, start=start)
            elif (self.ttl == 0 and self.state == 0) or (self.ttl > 0 and self.state == 1):
                self.state = 1
                if self.ttl == 0: 
                    self.ttl = max(1, int(np.random.exponential(scale=self.corr_params["event_scale"], size=None)))
                    self.sticky_value = None
                res = self.generate_correlated_transactions(num_txns, probabilities=probabilities, start=start, p=self.corr_params["correlation_factor"])
            else:
                assert False, "unreachable code"
            self.ttl -= 1
            return res

    def generate_random_transactions(self, num_txns:int, probabilities:list[float]=None, start=0) -> list[Transaction]:
        # start argument is used to deduplicate transaction ids when multiple time steps are scheduled

        if probabilities is None: probabilities = [1/len(self.SMALLBANK_TXN_OPS)] * len(self.SMALLBANK_TXN_OPS) # assume by default uniform

        txn_type_arr = random.choices(population=list(self.SMALLBANK_TXN_OPS.keys()), weights=probabilities, k=num_txns)
        
        txn_arr = []
        
        for i, v in enumerate(txn_type_arr):
            txn_id = start + i
            op_list = []
            cache = dict()
            for op in self.SMALLBANK_TXN_OPS[v]:
                val = cache[op[2]] if op[2] in cache else self.gen(self.SMALLBANK_INPUT_TYPINGS[v][op[2]])
                if op[2] not in cache: cache[op[2]] = val
                if op[0]: op_list.append(WriteOperation(txn_id, (op[1], val))) # write operation
                else: op_list.append(ReadOperation(txn_id, (op[1], val))) # read operation
            txn_arr.append(Transaction(txn_id, op_list))

        return txn_arr

    def generate_correlated_transactions(self, num_txns:int, probabilities:list[float]=None, start=0, p=0.1) -> list[Transaction]:
        # start argument is used to deduplicate transaction ids when multiple time steps are scheduled

        if probabilities is None: probabilities = [1/len(self.SMALLBANK_TXN_OPS)] * len(self.SMALLBANK_TXN_OPS) # assume by default uniform

        txn_type_arr = random.choices(population=list(self.SMALLBANK_TXN_OPS.keys()), weights=probabilities, k=num_txns)
        
        txn_arr = []
        
        for i, v in enumerate(txn_type_arr):
            txn_id = start + i
            op_list = []
            cache = dict()
            for op in self.SMALLBANK_TXN_OPS[v]:
                val = None
                if op[2] in cache:
                    val = cache[op[2]] # no choice, must take the cache value
                elif self.sticky_value is not None and random.random() < p: # for correlatedness
                    val = self.sticky_value
                else:
                    val = self.gen(self.SMALLBANK_INPUT_TYPINGS[v][op[2]])
                    if self.sticky_value is None: self.sticky_value = val
                if op[2] not in cache: cache[op[2]] = val
                if op[0]: op_list.append(WriteOperation(txn_id, (op[1], val))) # write operation
                else: op_list.append(ReadOperation(txn_id, (op[1], val))) # read operation
            txn_arr.append(Transaction(txn_id, op_list))
        return txn_arr

if __name__ == "__main__":
    workload = SmallBankWorkload()

    probabilities = [0.15, 0.15, 0.15, 0.25, 0.15, 0.15] # https://github.com/cmu-db/benchbase/blob/main/config/mysql/sample_smallbank_config.xml#L22
    num_txns = 100
    random_transactions = workload.generate_random_transactions(num_txns, probabilities)
    print(random_transactions)
   
