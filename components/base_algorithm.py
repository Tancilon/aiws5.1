from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def infer(self, **kwargs):
        pass
    
    @abstractmethod
    def build_cmd(self, **kwargs):
        pass