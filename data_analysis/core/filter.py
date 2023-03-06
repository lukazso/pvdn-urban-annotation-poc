from abc import ABC, abstractmethod

from typing import List, Tuple
import numpy as np
from .meta import AnnotatedObject, Difficulty, Distance, Scene, ImageAnnotation, ImageInfo, Vehicle


class BaseFilter(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def forward(self, scenes: List[Scene], annots: List[ImageAnnotation], img_infos: List[ImageInfo]) \
        -> Tuple[List[Scene], List[ImageAnnotation], List[ImageInfo]]:
        pass

    def __call__(self, scenes: List[Scene], annots: List[ImageAnnotation], img_infos: List[ImageInfo]):
        return self.forward(scenes, annots, img_infos)


class NumAnnotatorFilter(BaseFilter):
    def __init__(self, n: int) -> None:
        self.n = n
        super().__init__()
    
    def forward(self, scenes: List[Scene], annots: List[ImageAnnotation], img_infos: List[ImageInfo]) -> Tuple[List[Scene], List[ImageAnnotation], List[ImageInfo]]:
        filtered_scenes = []
        filtered_annots = []
        filtered_img_infos = []
        for scene in scenes:
            if len(scene.img_infos[0].annots) >= self.n:
                filtered_scenes.append(scene)
                filtered_annots += [a for img_info in scene.img_infos for a in img_info.annots]
                filtered_img_infos += scene.img_infos
    
        return filtered_scenes, filtered_annots, filtered_img_infos
