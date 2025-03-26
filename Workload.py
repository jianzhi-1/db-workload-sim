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
    def __init__(self) -> None:
        pass

    @abstractmethod
    def generate_random_transactions(self, probabilities:list[float], num_txns:int) -> list[Transaction]:
        ...

class SmallBankWorkload(Workload):
    def __init__(self) -> None:

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

    def gen(self, typ:str) -> int:
        if typ in ["long", "string"]: return draw_zipf_with_max(1.01, self.NUM_ACCOUNTS)
        elif typ in "double": return random.randint(0, 100)
        assert False, f"Error: type {typ} not supported in gen"

    def generate_random_transactions(self, num_txns:int, probabilities:list[float]=None) -> list[Transaction]:

        if probabilities is None: probabilities = [1/len(self.SMALLBANK_TXN_OPS)] * len(self.SMALLBANK_TXN_OPS) # assume by default uniform

        txn_type_arr = random.choices(population=list(self.SMALLBANK_TXN_OPS.keys()), weights=probabilities, k=num_txns)

        txn_arr = [Transaction(i, [
            WriteOperation(i, (op[1], self.gen(self.SMALLBANK_INPUT_TYPINGS[v][op[2]]))) 
                if op[0] else 
            ReadOperation(i, (op[1], self.gen(self.SMALLBANK_INPUT_TYPINGS[v][op[2]]))) 
            for op in self.SMALLBANK_TXN_OPS[v]
        ], v) for i, v in enumerate(txn_type_arr)]

        return txn_arr

if __name__ == "__main__":
    workload = SmallBankWorkload()

    probabilities = [0.15, 0.15, 0.15, 0.25, 0.15, 0.15] # https://github.com/cmu-db/benchbase/blob/main/config/mysql/sample_smallbank_config.xml#L22
    num_txns = 100
    random_transactions = workload.generate_random_transactions(num_txns, probabilities)
    print(random_transactions)
   
