from typing import List, Dict
import numpy as np

from data_analysis.core.meta import ReflectionType

from data_analysis.segmentation.utils import apply_segmentation
from data_analysis.segmentation.methods import SegmentationMethod
from data_analysis.segmentation.aggregation import Aggregator


class SegmentationPipeline:
    """
    A pipeline that applies a segmentation method to a set of boxes.

    Attributes:
        preproc (List): A list of preprocessing steps to apply before segmentation.
        method (SegmentationMethod): The segmentation method to apply to the boxes.
        postproc (List): A list of postprocessing steps to apply after segmentation.
    """
    def __init__(self, method: SegmentationMethod, preproc: List, postproc: List) -> None:
        """
        Constructs a SegmentationPipeline object.

        Args:
            method (SegmentationMethod): The segmentation method to apply to the boxes.
            preproc (List): A list of preprocessing steps to apply before segmentation.
            postproc (List): A list of postprocessing steps to apply after segmentation.
        """
        self.preproc = preproc
        self.method = method
        self.postproc = postproc
    
    def __call__(self, img: np.ndarray, boxes: List) -> np.ndarray:
        """
        Applies the segmentation method to the given boxes in the given image.

        Args:
            img (np.ndarray): The image to apply segmentation to.
            boxes (List): A list of boxes to apply segmentation to.
        
        Returns:
            np.ndarray: The resulting boolean mask of the segmentation operation, of shape (H, W), where H and W are the height and width of the image.
        """
        return apply_segmentation(img, boxes, self.method, self.preproc, self.postproc)



class SingleAnnotatorSegmentationPipeline:
    """
    A pipeline that applies a segmentation method to a set of boxes from a single annotator.

    Attributes:
        pipelines (Dict): A dictionary mapping reflection types to SegmentationPipeline objects.
    """
    def __init__(self, pipelines: Dict[ReflectionType, SegmentationPipeline]) -> None:
        """
        Constructs a SingleAnnotatorSegmentationPipeline object.

        Args:
            pipelines (Dict): A dictionary mapping reflection types to SegmentationPipeline objects.
        """
        self.pipelines = pipelines

    def apply_to_single_annotator(self, img: np.ndarray, boxes: List, types: List[ReflectionType]) -> np.ndarray:
        """
        Applies the segmentation method to the given boxes from a single annotator in the given image.

        Args:
            img (np.ndarray): The image to apply segmentation to.
            boxes (List): A list of boxes to apply segmentation to.
            types (List[ReflectionType]): A list of reflection types corresponding to each box. len(boxes) == len(types)
        
        Returns:
            np.ndarray: The resulting boolean mask of the segmentation operation, of shape (H, W), where H and W are the height and width of the image. 
        """
        assert len(boxes) == len(types)
        if len(boxes) == 0:
            mask = np.zeros(img.shape[:2], dtype=bool)
            return mask

        masks = []
        for box, tpe in zip(boxes, types):
            pipeline = self.pipelines[tpe]
            mask = pipeline(img, box)
            masks.append(mask)
        mask = np.logical_or.reduce(masks)
        return mask    

class MultiAnnotatorSegmentationPipeline(SingleAnnotatorSegmentationPipeline):
    """
    A pipeline that applies a segmentation method to a set of boxes from multiple annotators
    Args:
        pipeline (Dict[ReflectionType, SegmentationPipeline]): A dictionary mapping ReflectionType to SegmentationPipeline to be applied to each annotator's bounding boxes.
        aggregator (Aggregator): An aggregator to combine the masks from multiple annotators.

    Attributes:
        pipelines (Dict[ReflectionType, SegmentationPipeline]): A dictionary mapping ReflectionType to SegmentationPipeline to be applied to each annotator's bounding boxes.
        aggregator (Aggregator): An aggregator to combine the masks from multiple annotators.
    """

    def __init__(self, pipeline: Dict[ReflectionType, SegmentationPipeline], aggregator: Aggregator) -> None:
        """
        Initializes MultiAnnotatorSegmentationPipeline.

        Args:
            pipeline (Dict[ReflectionType, SegmentationPipeline]): A dictionary mapping ReflectionType to SegmentationPipeline to be applied to each annotator's bounding boxes.
            aggregator (Aggregator): An aggregator to combine the masks from multiple annotators.
        """
        super().__init__(pipeline)
        self.aggregator = aggregator

    def apply_to_multiple_annotators(self, img, boxes, types):
        """
        Applies SegmentationPipeline to the bounding boxes of multiple annotators and aggregates the resulting masks.

        Args:
            img (np.ndarray): Image to apply segmentation to.
            boxes (List[List[Box]]): List of bounding boxes, where each element is a list of Box objects from one annotator.
            types (List[List[ReflectionType]]): List of reflection types, where each element is a list of ReflectionType objects from one annotator.

        Returns:
            mask (np.ndarray): Mask of shape (H, W) where H and W are the height and width of the image.
        """
        assert len(boxes) == len(types)

        masks = []
        for annotator_boxes, annotator_types in zip(boxes, types):
            mask = self.apply_to_single_annotator(img, annotator_boxes, annotator_types)
            masks.append(mask)

        mask = self.aggregator(masks)
        return mask
