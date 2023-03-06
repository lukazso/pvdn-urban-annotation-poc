from abc import ABC, abstractmethod
from typing import List
import numpy as np


class Aggregator(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def forward(self, masks: List[np.ndarray]):
        pass

    def __call__(self, masks: List[np.ndarray]):
        assert isinstance(masks, list)
        return self.forward(masks)

    
class UnionAggregator(Aggregator):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, masks: List[np.ndarray]):
        union_mask = np.logical_or.reduce(masks)
        return union_mask


class IntersectionAggregator(Aggregator):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, masks: List[np.ndarray]):
        intersection_mask = np.logical_and.reduce(masks)
        return intersection_mask


class MajorityAggregator(Aggregator):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, masks: List[np.ndarray]):
        majority_mask = np.sum(masks, axis=0) > len(masks) // 2
        return majority_mask