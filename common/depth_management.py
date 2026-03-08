from abc import ABC, abstractmethod
from pathlib import Path


class DepthManagement(ABC):
    def __init__(self):
        super().__init__
    
    @abstractmethod
    def check_depth(depth_path: Path):
        raise NotImplementedError()