from abc import ABC, abstractmethod
from utils import Transaction, ReadOperation, WriteOperation, InputTyping
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
        
        self.SMALLBANK_TXN_OPS = { # (Read/Write, Table ID, column_name, cache-id)
            # Operations with the same cache-id operate on the same row
            # False -> Read, True -> Write
            'Amalgamate': [(False, 0, 'custId0', 0), (False, 0, 'custId1', 1), (False, 1, 'custId0', 2), (False, 2, 'custId1', 3), (True, 2, 'custId0', 4), (True, 2, 'custId1', 5)],
            'Balance': [(False, 0, 'custName', 0), (False, 1, 'custName', 1), (False, 2, 'custName', 2)],
            'DepositChecking': [(False, 0, 'custName', 0), (True, 0, 'custName', 0)],
            'SendPayment': [(False, 0, 'sendAcct', 0), (False, 0, 'destAcct', 1), (False, 2, 'sendAcct', 2), (True, 2, 'sendAcct', 2), (True, 2, 'destAcct', 3)],
            'TransactSavings': [(False, 0, 'custName', 0), (False, 1, 'custName', 1), (True, 1, 'custName', 1)],
            'WriteCheck': [(False, 0, 'custName', 0), (False, 1, 'custName', 1), (False, 2, 'custName', 2), (True, 2, 'custName', 2)],
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

    def get_sticky_list(self) -> list:
        if self.state == 0: return None
        if self.sticky_value is None: return None
        return [(i, self.sticky_value) for i in range(3)]

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
                val = cache[op[3]] if op[3] in cache else self.gen(self.SMALLBANK_INPUT_TYPINGS[v][op[2]])
                if op[3] not in cache: cache[op[3]] = val
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
                if op[3] in cache:
                    val = cache[op[3]] # no choice, must take the cache value
                elif self.sticky_value is not None and random.random() < p: # for correlatedness
                    val = self.sticky_value
                else:
                    val = self.gen(self.SMALLBANK_INPUT_TYPINGS[v][op[2]])
                    if self.sticky_value is None: self.sticky_value = val
                if op[3] not in cache: cache[op[3]] = val
                if op[0]: op_list.append(WriteOperation(txn_id, (op[1], val))) # write operation
                else: op_list.append(ReadOperation(txn_id, (op[1], val))) # read operation
            txn_arr.append(Transaction(txn_id, op_list))
        return txn_arr

class TPCCWorkload(Workload):
    def __init__(self, correlated:bool=False, corr_params:dict=None) -> None:
        self.NUM_WAREHOUSES = 10
        self.NUM_DISTRICTS = 10
        self.NUM_CUSTOMERS = 3000
        self.NUM_ORDERS = 3000
        self.NUM_ITEMS = 100000

        self.TPCC_OPS = {
            'Delivery': [# loop start 10 
                #(isWrite, table, is_last_on_resource, where clauses)
                (False, 'N', False, ('D_ID', 'O_ID', 'W_ID')),
                (True, 'N', True, ('D_ID', 'O_ID', 'W_ID')),
                (False, 'O', False, ('D_ID', 'O_ID', 'W_ID')),
                (True, 'O', True, ('D_ID', 'O_ID', 'W_ID')),
                (True, 'L', False, ('D_ID', 'O_ID', 'W_ID')),
                (False, 'L', True, ('D_ID', 'O_ID', 'W_ID')),
                (True, 'C', True, ('C_ID', 'D_ID', 'W_ID')), # loop end
            ],
            'NewOrder': [
                (False, 'C', True, ('C_ID', 'D_ID', 'W_ID')),
                (False, 'W', True, ('W_ID',)),
                (False, 'D', False, ('D_ID', 'W_ID')),
                (True, 'D', True, ('D_ID', 'W_ID')),
                (False, 'I', True, ('D_ID', 'I_ID', 'O_ID', 'W_ID')),  # loop start, stmtInsertOOrder
                (True, 'S', True, ('I_ID', 'W_ID')),  # loop end
                # Batch updates which insert new rows
            ],
            'OrderStatus_ID': [
                (False, 'C', True, ('C_ID', 'D_ID', 'W_ID')), #getCustomerByID
                (False, 'O', True, ('C_ID', 'D_ID', 'W_ID')),
                (False, 'L', True, ('D_ID', 'O_ID', 'W_ID')),
            ],
            'OrderStatus_NAME': [
                (False, 'C', True, ('D_ID', 'LAST', 'W_ID')), #getCustomerByName
                (False, 'O', True, ('C_ID', 'D_ID', 'W_ID')),
                (False, 'L', True, ('D_ID', 'O_ID', 'W_ID')),
            ],
            'Payment1_ID': [
                (True, 'W', True, ('W_ID',)),
                (False, 'W', True, ('W_ID',)),
                (True, 'D', False, ('D_ID', 'W_ID')),
                (False, 'D', True, ('D_ID', 'W_ID')),
                (False, 'C', False, ('C_ID', 'D_ID', 'W_ID')),
                (False, 'C', False, ('C_ID', 'D_ID', 'W_ID')),
                (True, 'C', True, ('C_ID', 'D_ID', 'W_ID')),
            ],
            'Payment1_NAME': [
                (True, 'W', True, ('W_ID',)),
                (False, 'W', True, ('W_ID',)),
                (True, 'D', False, ('D_ID', 'W_ID')),
                (False, 'D', True, ('D_ID', 'W_ID')),
                (False, 'C', True, ('D_ID', 'LAST', 'W_ID')),
                (False, 'C', False, ('C_ID', 'D_ID', 'W_ID')),
                (True, 'C', True, ('C_ID', 'D_ID', 'W_ID')),
            ],
            'Payment2_ID': [
                (True, 'W', True, ('W_ID',)),
                (False, 'W', True, ('W_ID',)),
                (True, 'D', False, ('D_ID', 'W_ID')),
                (False, 'D', True, ('D_ID', 'W_ID')),
                (False, 'C', False, ('C_ID', 'D_ID', 'W_ID')),
                (True, 'C', True, ('C_ID', 'D_ID', 'W_ID')),
            ],
            'Payment2_NAME': [
                (True, 'W', True, ('W_ID',)),
                (False, 'W', True, ('W_ID',)),
                (True, 'D', False, ('D_ID', 'W_ID')),
                (False, 'D', True, ('D_ID', 'W_ID')),
                (False, 'C', True, ('D_ID', 'LAST', 'W_ID')), #getCustomerByN
                (True, 'C', True, ('C_ID', 'D_ID', 'W_ID')),

            ],
            'StockLevel': [
                (False, 'D', True, ('D_ID', 'W_ID')),
                (False, 'L', True, ('D_ID', 'O_ID', 'W_ID'))
            ]
        }

        self.TPCC_InputTypings = {
            "C_ID" : InputTyping('C_ID', 1, self.NUM_CUSTOMERS, random.randint(0, 1023)),
            "D_ID" : InputTyping('int', 1, self.NUM_DISTRICTS),
            "I_ID" : InputTyping('I_ID', 1, self.NUM_ITEMS, random.randint(0, 8191)),
            "LAST" : InputTyping('LAST', 0, 999, random.randint(0, 255)),
            "O_ID" : InputTyping('int', 1, self.NUM_ORDERS),
            "W_ID" : InputTyping('int', 1, self.NUM_WAREHOUSES),
        }

        self.TPCC_TableMaps = {
            # C, D, I, L, O, S, W
            'C': 0, #CUSTOMER
            'D': 1, #District
            'I': 2, #Item
            'L': 3, #ORDERLINE
            'O': 4, #OPENORDER
            'S': 5, #Stock
            'W': 6, #Warehouse
            'N': 7, #NEWORDER
        }

        self.num_txn_types = len(self.TPCC_OPS)

        #self.SMALLBANK_HOT_KEYS = [(0,i) for i in range(10)] + [(1, i) for i in range(10)] + [(2, i) for i in range(10)] # ???
        self.TPCC_HOT_KEYS = [(6,i) for i in range(self.NUM_WAREHOUSES)] + [(1, i) for i in range(self.NUM_DISTRICTS)] 

        self.correlated:bool = correlated
        self.corr_params:dict = corr_params
        self.ttl = None # time to next event
        self.state = 0 # 0: random, 1: correlated
        self.sticky_value = None # the value that is correlated to during that time interval
    
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
        
    #def generate_random_transactions(self, num_txns:int, probabilities:list[float]=None, start=0) -> list[Transaction]:
        #return [self.TPCC_generate_random_transaction(probabilities, start+i) for i in range(num_txns)]

    def generate_random_transactions(self, num_txns:int, probabilities:list[float]=None, start:int=0):
        txn_ops = self.TPCC_OPS
        if probabilities is None:
            probabilities = [1/len(txn_ops)] * len(txn_ops)
        num_txn_types = len(txn_ops.keys())
        txn_input_typings = self.TPCC_InputTypings

        txn_arr = []
        for i in range(num_txns):
            txn_type_id = random.choices(population=([i for i in range(0, num_txn_types)]), weights=probabilities, k=1)[0]
            txn_type = list(txn_ops.keys())[txn_type_id]

            ops_unformatted = txn_ops[txn_type]
            num_reps = 1
            if txn_type == "NewOrder":
                num_reps = random.randint(5, 10)
            result_ops = []

            for _ in range(num_reps):
                my_inputs = {}

                for key in txn_input_typings.keys():
                    cur_input_typing = txn_input_typings[key]
                    my_inputs[key] = cur_input_typing.generate_value()

                for op_idx in range(len(ops_unformatted)):  # Iterate over the sequence of ops:
                    isWrite, table, is_last_on_resource, rows = ops_unformatted[op_idx]
                    res_rows = {}
                    for row in rows:
                        res_rows[row] = my_inputs[row]
                    res =  ','.join(f"{k}:{v}" for k, v in res_rows.items())

                    if not isWrite: #read
                        result_ops.append(ReadOperation(start+i, (self.TPCC_TableMaps[table],res), False, is_last_on_resource, res_rows))
                    else: #write
                        result_ops.append(WriteOperation(start+i, (self.TPCC_TableMaps[table],res), False, is_last_on_resource, res_rows))
            
            result_ops[-1].is_last = True
            txn = Transaction(start+i, result_ops, txn_type)
            txn_arr.append(txn)
        return txn_arr
    
    def generate_correlated_transactions(self, num_txns:int, probabilities:list[float]=None, start:int=0, p:float=0.1):
        txn_ops = self.TPCC_OPS
        if probabilities is None:
            probabilities = [1/len(txn_ops)] * len(txn_ops)
        num_txn_types = len(txn_ops.keys())

        txn_input_typings = self.TPCC_InputTypings
        txn_arr = []
        #sticky values
        sticky_inputs = {}
        for key in txn_input_typings.keys():
            cur_input_typing = txn_input_typings[key]
            sticky_inputs[key] = cur_input_typing.generate_value()

        for i in range(num_txns):
            txn_type_id = random.choices(population=([i for i in range(0, num_txn_types)]), weights=probabilities, k=1)[0]
            txn_type = list(txn_ops.keys())[txn_type_id]

            ops_unformatted = txn_ops[txn_type]
            result_ops = []

            num_reps = 1
            if txn_type == "NewOrder":
                num_reps = random.randint(5, 10)
            result_ops = []

            for _ in range(num_reps):
                my_inputs = {}

                for key in txn_input_typings.keys():
                    cur_input_typing = txn_input_typings[key]
                    if random.random() < p:
                        my_inputs[key] = sticky_inputs[key]
                    else:
                        my_inputs[key] = cur_input_typing.generate_value()

                for op_idx in range(len(ops_unformatted)):  # Iterate over the sequence of ops:
                    isWrite, table, is_last_on_resource, rows = ops_unformatted[op_idx]

                    res_rows = {}
                    for row in rows:
                        res_rows[row] = my_inputs[row]
                    res =  ','.join(f"{k}:{v}" for k, v in res_rows.items())

                    if not isWrite: #read
                        result_ops.append(ReadOperation(start+i, (self.TPCC_TableMaps[table],res), False, is_last_on_resource, res_rows))
                    else: #write
                        result_ops.append(WriteOperation(start+i, (self.TPCC_TableMaps[table],res), False, is_last_on_resource, res_rows))
            
            result_ops[-1].is_last = True
            txn = Transaction(start+i, result_ops, txn_type)
            txn_arr.append(txn)
        return txn_arr

class YCSBWorkload(Workload):
    def __init__(self, correlated:bool=False, corr_params:dict=None) -> None:

        self.NUM_RECORDS = 1000
        self.MAX_SCAN = 1000

        self.YCSB_OPS = {  #(isWrite, table, is_last_on_resource, where clauses)
            'DeleteRecord' : [
                (True, 'U', True, ('YCSB_KEY',)),
            ],
            'InsertRecord' : [
                (True, 'U', True, ('YCSB_KEY',)),
            ],
            'ReadModifyWriteRecord' : [
                (False, 'U', False, ('YCSB_KEY',)),
                (True, 'U', True, ('YCSB_KEY',)),
            ],
            'ReadRecord' : [
                (False, 'U', True, ('YCSB_KEY',)),
            ],
            'ScanRecord' : [ # range of lower to upper bound
                (False, 'U', False, ('YCSB_KEY',)),
            ],
            'UpdateRecord' : [
                (True, 'U', True, ('YCSB_KEY',)),
            ],            
        }

        self.YCSB_InputTypings = {
            "YCSB_KEY" : InputTyping('zipf', 1, self.NUM_RECORDS, None),
        }

        self.YCSB_TableMaps = {
            'U': 0, #USERTABLE
        }

        self.num_txn_types = len(self.YCSB_OPS)

        self.YCSB_HOT_KEYS = [(0,i) for i in range(10)] # first 10 YCSB_KEYs are hot

        self.correlated:bool = correlated
        self.corr_params:dict = corr_params
        self.ttl = None # time to next event
        self.state = 0 # 0: random, 1: correlated
        self.sticky_value = None # the value that is correlated to during that time interval
    
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
        
    #def generate_random_transactions(self, num_txns:int, probabilities:list[float]=None, start=0) -> list[Transaction]:
        #return [self.TPCC_generate_random_transaction(probabilities, start+i) for i in range(num_txns)]

    def generate_random_transactions(self, num_txns:int, probabilities:list[float]=None, start:int=0):
        txn_ops = self.YCSB_OPS
        if probabilities is None:
            probabilities = [1/len(txn_ops)] * len(txn_ops)
        num_txn_types = len(txn_ops.keys())
        txn_input_typings = self.YCSB_InputTypings

        txn_arr = []
        for i in range(num_txns):
            txn_type_id = random.choices(population=([i for i in range(0, num_txn_types)]), weights=probabilities, k=1)[0]
            txn_type = list(txn_ops.keys())[txn_type_id]

            ops_unformatted = txn_ops[txn_type]
            result_ops = []

            my_inputs = {}

            for key in txn_input_typings.keys():
                cur_input_typing = txn_input_typings[key]
                my_inputs[key] = cur_input_typing.generate_value()

            row = my_inputs['YCSB_KEY']

            if txn_type == "ScanRecord":
                scan_count = random.randint(1, self.MAX_SCAN)
                if row + scan_count > self.NUM_RECORDS:
                    scan_count = self.NUM_RECORDS - row

                for scan_idx in range(scan_count):
                    result_ops.append(ReadOperation(start+i, (self.YCSB_TableMaps[table],row+scan_idx), False, is_last_on_resource))
            else:
                for op_idx in range(len(ops_unformatted)):  # Iterate over the sequence of ops:
                    isWrite, table, is_last_on_resource, rows = ops_unformatted[op_idx]

                    if not isWrite: #read
                        result_ops.append(ReadOperation(start+i, (self.YCSB_TableMaps[table],row), False, is_last_on_resource))
                    else: #write
                        result_ops.append(WriteOperation(start+i, (self.YCSB_TableMaps[table],row), False, is_last_on_resource))
            
            result_ops[-1].is_last = True
            txn = Transaction(start+i, result_ops, txn_type)
            txn_arr.append(txn)
        return txn_arr
    
    def generate_correlated_transactions(self, num_txns:int, probabilities:list[float]=None, start:int=0, p:float=0.1):
        txn_ops = self.YCSB_OPS
        if probabilities is None:
            probabilities = [1/len(txn_ops)] * len(txn_ops)
        num_txn_types = len(txn_ops.keys())

        txn_input_typings = self.YCSB_InputTypings
        txn_arr = []
        #sticky values
        sticky_inputs = {}
        for key in txn_input_typings.keys():
            cur_input_typing = txn_input_typings[key]
            sticky_inputs[key] = cur_input_typing.generate_value()

        for i in range(num_txns):
            txn_type_id = random.choices(population=([i for i in range(0, num_txn_types)]), weights=probabilities, k=1)[0]
            txn_type = list(txn_ops.keys())[txn_type_id]

            ops_unformatted = txn_ops[txn_type]
            result_ops = []

            my_inputs = {}

            for key in txn_input_typings.keys():
                cur_input_typing = txn_input_typings[key]
                if random.random() < p:
                    my_inputs[key] = sticky_inputs[key]
                else:
                    my_inputs[key] = cur_input_typing.generate_value()
            
            row = my_inputs['YCSB_KEY']

            if txn_type == "ScanRecord":
                scan_count = random.randint(1, self.MAX_SCAN)
                if row + scan_count > self.NUM_RECORDS:
                    scan_count = self.NUM_RECORDS - row

                for scan_idx in range(scan_count):
                    result_ops.append(ReadOperation(start+i, (self.YCSB_TableMaps[table],row+scan_idx), False, is_last_on_resource))
            else:
                for op_idx in range(len(ops_unformatted)):  # Iterate over the sequence of ops:
                    isWrite, table, is_last_on_resource, rows = ops_unformatted[op_idx]

                    if not isWrite: #read
                        result_ops.append(ReadOperation(start+i, (self.YCSB_TableMaps[table],row), False, is_last_on_resource))
                    else: #write
                        result_ops.append(WriteOperation(start+i, (self.YCSB_TableMaps[table],row), False, is_last_on_resource))
            
            result_ops[-1].is_last = True
            txn = Transaction(start+i, result_ops, txn_type)
            txn_arr.append(txn)
        return txn_arr



if __name__ == "__main__":
    workload = YCSBWorkload()

    #probabilities = [0.15, 0.15, 0.15, 0.25, 0.15, 0.15] # https://github.com/cmu-db/benchbase/blob/main/config/mysql/sample_smallbank_config.xml#L22
    num_txns = 100
    random_transactions = workload.generate_random_transactions(num_txns)
    #print(random_transactions)
    for txn in random_transactions:
        print(txn)
   
