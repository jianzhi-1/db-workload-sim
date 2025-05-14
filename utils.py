from typing import Union
from random import randint as randint # TPCC

# TPCC
class InputTyping:
     def __init__(self, input_type, range_min, range_max):
         self.input_type = input_type
         self.range_min = range_min
         self.range_max = range_max
     
     def generate_value(self):
         return randint(self.range_min, self.range_max)

class Operation():
    def __init__(self, txn:int, resource:str):
        self.is_read = None
        self.txn = txn
        self.type = "X"
        self.resource = resource
        self.is_last = False
        self.is_last_on_resource = False
        self.delta_last = -1 # the number of operations from it to the last operation on this resource

    def get_resource(self):
        return self.resource

class ReadOperation(Operation):
    def __init__(self, txn:int, resource:Union[str, tuple], is_last:bool=False, is_last_on_resource:bool=False, res_rows:dict=None):
        super().__init__(txn, resource)
        self.is_read = True
        self.type = "R"
        self.is_last = is_last
        self.is_last_on_resource = is_last_on_resource
        self.delta_last = -1
        self.res_rows = res_rows # TPCC

    def __repr__(self):
        return f"Read({self.resource})"

class WriteOperation(Operation):
    def __init__(self, txn:int, resource:Union[str, tuple], is_last:bool=False, is_last_on_resource:bool=False, res_rows:dict=None):
        super().__init__(txn, resource)
        self.is_read = False
        self.type = "W"
        self.is_last = is_last
        self.is_last_on_resource = is_last_on_resource
        self.delta_last = -1
        self.res_rows = res_rows # TPCC

    def __repr__(self):
        return f"Write({self.resource})"

class Transaction():
    def __init__(self, txn:int, operations:list[Operation], txn_type:str="", resource_to_ts:dict={}):
        self.txn = txn
        self.txn_type = txn_type # annotation only, not used
        assert len(operations) > 0
        self.operations = operations  # List of Read/Write objects
        self.operations[-1].is_last = True
        self.resource_to_ts = resource_to_ts #mapping of resource to timestep, hopefully to reduct computation on conflict matrix for checking conflict resources
        visited = dict()
        for i in range(len(self.operations) - 1, -1, -1):
            if self.operations[i].resource not in visited:
                self.operations[i].is_last_on_resource = True
                self.operations[i].delta_last = 0 # it is the last operation
                visited[self.operations[i].resource] = i
            else:
                self.operations[i].delta_last = visited[self.operations[i].resource] - i

    def __repr__(self):
        # return f"Transaction({self.txn}, {self.operations}, {self.resource_to_ts})"
        return f"Transaction({self.txn}, {self.operations})"

def clone_operation_list(ls:list[Operation]):
    return [ReadOperation(x.txn, x.resource, x.is_last, x.is_last_on_resource, x.res_rows) if x.is_read else WriteOperation(x.txn, x.resource, x.is_last, x.is_last_on_resource, x.res_rows) for x in ls]

def clone_transaction(T:Transaction) -> Transaction:
    return Transaction(T.txn, clone_operation_list(T.operations), txn_type=T.txn_type)

def conflict(T1:Transaction, T2:Transaction, t:int) -> bool:
    """
    Given two transactions T1 and T2 with T2 being t time steps after T1,
    return whether the two transactions will conflict.
    """
    if t < 0: return conflict(T2, T1, -t)
    T1beforeT2, T2beforeT1 = False, False
    resource_lock = dict()

    def type_to_num(op):
        return 1 if op.is_read else 2

    for i in range(t + max(len(T1.operations), len(T2.operations))):
        if i < len(T1.operations):
            op = T1.operations[i]
            if op.resource in resource_lock:
                a, b = resource_lock[op.resource]
                if b > 0 and max(type_to_num(op), b) == 2: # add edge if one of them is a write
                    # actually, there should also be a condition b > 0, but degrades performance
                    # T2beforeT1 = True
                    return True
                resource_lock[op.resource] = (max(type_to_num(op), a), b)
            else:
                resource_lock[op.resource] = (type_to_num(op), 0)
            if op.is_last_on_resource:
                a, b = resource_lock[op.resource]
                resource_lock[op.resource] = (0, b)
        if 0 <= (i - t) and (i - t) < len(T2.operations):
            op = T2.operations[i - t]
            if op.resource in resource_lock:
                a, b = resource_lock[op.resource]
                if a > 0 and max(a, type_to_num(op)) == 2: # add edge if one of them is a write
                    # actually, there should also be a condition a > 0, but degrades performance
                    # T1beforeT2 = True
                    return True
                resource_lock[op.resource] = (a, max(type_to_num(op), b))
            else:
                resource_lock[op.resource] = (0, type_to_num(op))
            if op.is_last_on_resource:
                a, b = resource_lock[op.resource]
                resource_lock[op.resource] = (a, 0)
        
        # if (T1beforeT2 and T2beforeT1):
            # return True

    # return T1beforeT2 and T2beforeT1
    return False

    