from abc import ABC, abstractmethod
from utils import Transaction

class Scheduler(ABC):

    @abstractmethod
    def schedule(self, Wt:dict(), Locker, TxnPool:list[Transaction]) -> list[int]:
        ...

class RapidFireScheduler(Scheduler):
    def schedule(self, Wt:dict(), Locker, TxnPool:list[Transaction]) -> list[int]:
        total_txns = len(TxnPool)
        assert total_txns > 0, "scheduler fail: total number of transactions is 0"
        return [1] + [0]*(total_txns - 1)

class SequentialScheduler(Scheduler):
    def schedule(self, Wt:dict(), Locker, TxnPool:list[Transaction]) -> list[int]:
        total_txns = len(TxnPool)
        assert total_txns > 0, "scheduler fail: total number of transactions is 0"
        if len(Wt) > 0: return [0]*total_txns
        return [1] + [0]*(total_txns - 1)