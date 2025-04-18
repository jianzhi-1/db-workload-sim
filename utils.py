from typing import Union

class Operation():
    def __init__(self, txn:int, resource:str):
        self.is_read = None
        self.txn = txn
        self.type = "X"
        self.resource = resource
        self.is_last = False
        self.is_last_on_resource = False

    def get_resource(self):
        return self.resource

class ReadOperation(Operation):
    def __init__(self, txn:int, resource:Union[str, tuple], is_last:bool=False, is_last_on_resource:bool=False):
        super().__init__(txn, resource)
        self.is_read = True
        self.type = "R"
        self.is_last = is_last
        self.is_last_on_resource = is_last_on_resource

    def __repr__(self):
        return f"Read({self.resource})"

class WriteOperation(Operation):
    def __init__(self, txn:int, resource:Union[str, tuple], is_last:bool=False, is_last_on_resource:bool=False):
        super().__init__(txn, resource)
        self.is_read = False
        self.type = "W"
        self.is_last = is_last
        self.is_last_on_resource = is_last_on_resource

    def __repr__(self):
        return f"Write({self.resource})"

class Transaction():
    def __init__(self, txn:int, operations:list[Operation], txn_type:str=""):
        self.txn = txn
        self.txn_type = txn_type # annotation only, not used
        assert len(operations) > 0
        self.operations = operations  # List of Read/Write objects
        self.operations[-1].is_last = True
        visited = set()
        for i in range(len(self.operations) - 1, -1, -1):
            if self.operations[i].resource not in visited:
                self.operations[i].is_last_on_resource = True
                visited.add(self.operations[i].resource)

    def __repr__(self):
        return f"Transaction({self.txn}, {self.operations})"

def conflict(T1:Transaction, T2:Transaction, t:int):
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
                if max(type_to_num(op), b) == 2: # add edge if one of them is a read
                    T2beforeT1 = True
                resource_lock[op.resource] = (max(type_to_num(op), a), b)
            else:
                resource_lock[op.resource] = (type_to_num(op), 0)
            if op.is_last_on_resource:
                a, b = resource_lock[op.resource]
                resource_lock[op.resource] = (0, b)
        if i - t >= 0 and i - t < len(T2.operations):
            op = T2.operations[i - t]
            if op.resource in resource_lock:
                a, b = resource_lock[op.resource]
                if max(a, type_to_num(op)) == 2: # add edge if one of them is a write
                    T1beforeT2 = True
                resource_lock[op.resource] = (a, max(type_to_num(op), b))
            else:
                resource_lock[op.resource] = (0, type_to_num(op))
            if op.is_last_on_resource:
                a, b = resource_lock[op.resource]
                resource_lock[op.resource] = (a, 0)

    return T1beforeT2 and T2beforeT1

    