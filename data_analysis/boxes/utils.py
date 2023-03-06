from typing import List
import numpy as np

from data_analysis.core.meta import DirectVehicle, ImageAnnotation


from typing import List
import numpy as np

from data_analysis.core.meta import DirectVehicle, ImageAnnotation


def count_num_boxes(annots: ImageAnnotation, indirect: bool = True, direct: bool = False) -> int:
    """Counts the number of boxes in an image annotation object.
    
    Args:
        annots (ImageAnnotation): An object containing image annotation data.
        indirect (bool, optional): Whether to count boxes for indirect vehicles. Defaults to True.
        direct (bool, optional): Whether to count boxes for direct vehicles. Defaults to False.
    
    Raises:
        ValueError: If both indirect and direct are False.
    
    Returns:
        int: The total number of boxes counted.
    """
    if not indirect and not direct:
        raise ValueError("At least one of 'indirect' or 'direct' must be True.")
    
    num_direct = 0
    num_indirect = 0
    if indirect:
        num_indirect = sum([len(v.instances) for v in annots.vehicles])
    if direct:
        num_direct = sum([isinstance(v, DirectVehicle) for v in annots.vehicles])
    return num_indirect + num_direct


def jaccard(boxes: List) -> float:
    """Calculates the Jaccard index for a list of bounding boxes.
    
    Args:
        boxes (List): A list of bounding boxes in the format [xmin, ymin, xmax, ymax].
    
    Returns:
        float: The Jaccard index for the set of bounding boxes.
    """
    if len(boxes) < 2:
        raise ValueError("You need to provide at least 2 boxes.")
    # turn boxes into masks
    num_boxes = len(boxes)
    boxes = np.array(boxes)

    height = boxes[:, 3] - boxes[:, 1]
    width = boxes[:, 2] - boxes[:, 0]
    
    x1 = boxes[:, 0].min()
    x2 = boxes[:, 2].max()

    y1 = boxes[:, 1].min()
    y2 = boxes[:, 3].max()

    width = x2 - x1
    height = y2 - y1
    
    boxes[:, [0, 2]] -= x1
    boxes[:, [1, 3]] -= y1
    
    img = np.zeros((height, width))
    for box in boxes:
        x1, y1, x2, y2 = box
        img[y1:y2, x1:x2] += 1
    
    intersection = np.count_nonzero(img == num_boxes)
    union = np.count_nonzero(img > 0)

    iou = intersection / union
    return iou


if __name__ == "__main__":
    boxes = [
        [10, 10, 20, 20], 
        [10, 10, 15, 15]
    ]

    iou = jaccard(boxes)
    print(iou)