from abc import ABC, abstractmethod
from pathlib import Path


class MaskManagement(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def convert_mask(mask_path: Path):
        pass