from abc import ABC, abstractclassmethod
import numpy as np
import cv2

from data_analysis.segmentation.extensions.c_image_operations import Cblob
from data_analysis.segmentation.bms import compute_saliency

class SegmentationMethod(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    
    def __call__(self, img: np.ndarray):
        assert isinstance(img, np.ndarray)
        assert img.dtype == np.uint8
        return self.forward(img)
    
    @abstractclassmethod
    def forward(self, img: np.ndarray):
        pass
    

class SimpleThresholding(SegmentationMethod):
    def __init__(self, t: int) -> None:
        super().__init__()
        self.t = t
    
    def forward(self, img):
        mask = img > self.t
        return mask


class GlobalMeanThresholding(SegmentationMethod):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k
    
    def forward(self, img):
        t = np.mean(img) + self.k
        mask = img > t
        return mask


class AdaptiveMeanThresholding(SegmentationMethod):
    def __init__(self, w: int, c: int) -> None:
        super().__init__()
        self.w = w
        self.c = c
    
    def forward(self, img):
        mask = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=self.w, C=self.c)
        mask = mask.astype(bool)
        return mask


class AdaptiveGaussianThresholding(SegmentationMethod):
    def __init__(self, w: int, c: int) -> None:
        super().__init__()
        self.w = w
        self.c = c
    
    def forward(self, img):
        mask = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=self.w, C=self.c)
        mask = mask.astype(bool)
        return mask


class DynamicThresholding(SegmentationMethod):
    def __init__(self, k: float = 0.06, w: int = 11, eps: float = 1e-3) -> None:
        super().__init__()
        self.threshold = Cblob(k, w, eps)
    
    def forward(self, img):
        img = img.astype(float)
        img /= 255

        mask = self.threshold.binarize_in_c(img)
        mask = mask.astype(bool)
        return mask


class BMSThresholding(SegmentationMethod):
    def __init__(self, t: int, n_thresholds: int = 5) -> None:
        super().__init__()
        self.t = t
        self.n_thresholds = n_thresholds
    
    def forward(self, img):
        smap = compute_saliency(img, self.n_thresholds)     # this returns a np.uint8 image
        mask = smap > self.t
        return mask


class OtsuThresholding(SegmentationMethod):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, img):
        _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = mask.astype(bool)
        return mask
