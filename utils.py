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
    