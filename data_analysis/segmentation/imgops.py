from abc import ABC, abstractclassmethod
from typing import List, Tuple

import numpy as np
import cv2


class BasicImgOp(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, img):
        assert isinstance(img, np.ndarray)
        assert (img.dtype == np.uint8) or (img.dtype == bool)
        return self.forward(img)
    
    @abstractclassmethod
    def forward(self, img):
        pass


class NormalizeMinMax(BasicImgOp):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, img):
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return img


class Resize(BasicImgOp):
    def __init__(self, size: Tuple[int, int], interpolation: int = cv2.INTER_AREA) -> None:
        super().__init__()
        self.size = size
        self.interpolation = interpolation
    
    def forward(self, img):
        img = cv2.resize(img, self.size, interpolation=self.interpolation)
        return img


class ToGrayscale(BasicImgOp):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img


class GaussianBlur(BasicImgOp):
    def __init__(self, w: Tuple[int, int], sigma: int) -> None:
        super().__init__()
        self.w = w
        self.sigma = sigma
        
    def forward(self, img):
        img = cv2.GaussianBlur(img, self.w, self.sigma)
        return img


class MedianBlur(BasicImgOp):
    def __init__(self, ksize: int) -> None:
        super().__init__()
        self.ksize = ksize
    
    def forward(self, img):
        img = cv2.medianBlur(img, self.ksize)
        return img


class MeanBlur(BasicImgOp):
    def __init__(self, w: Tuple[int, int]) -> None:
        super().__init__()
        self.w = w
    
    def forward(self, img):
        img = cv2.blur(img, self.w)
        return img


class Erosion(BasicImgOp):
    def __init__(self, kernel: np.ndarray, iterations: int) -> None:
        super().__init__()
        self.kernel = kernel
        self.iterations = iterations
    
    def forward(self, img):
        img = cv2.erode(img, self.kernel, iterations=self.iterations)
        return img


class Dilation(BasicImgOp):
    def __init__(self, kernel: np.ndarray, iterations: int) -> None:
        super().__init__()
        self.kernel = kernel
        self.iterations = iterations
    
    def forward(self, img):
        img = cv2.dilate(img, self.kernel, iterations=self.iterations)
        return img
