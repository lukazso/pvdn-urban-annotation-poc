from typing import List
from abc import ABC, abstractmethod

import numpy as np


class BaseFilter(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    @abstractmethod
    def forward(self, data: List):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class EmptyFilter(BaseFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, data: List):
        data_filtered = []
        for container in data:
            mask = container.get_mask()
            if np.count_nonzero(mask) > 0:
                data_filtered.append(container)
        return data_filtered
