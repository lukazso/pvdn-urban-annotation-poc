from typing import List, Union
import numpy as np
import cv2

from data_analysis.core.meta import DirectVehicle, Reflection

from data_analysis.segmentation.methods import SegmentationMethod, BMSThresholding
from data_analysis.segmentation.imgops import BasicImgOp
from data_analysis.core.utils import COLORS


def jaccard(masks: List[np.ndarray]):
    num_masks = len(masks)
    joined_mask = np.array(masks, dtype=int)
    summed = joined_mask.sum(0)

    union = np.count_nonzero(summed > 0)
    if union == 0:
        return -1
    intersection = np.count_nonzero(summed == num_masks)
    iou = intersection / union
    
    return iou


def apply_segmentation(img: np.ndarray, box: List[int], method: SegmentationMethod, preproc: List[BasicImgOp], postproc: List[BasicImgOp]):
    """
    Returns:
        Boolean mask of same shape as as input image with the segmented bounding box area.
    """
    # cut out the bounding box from the image
    x1, y1, x2, y2 = box
    h_orig = y2 - y1
    w_orig = x2 - x1
    cutout = img[y1:y2, x1:x2, :]

    # apply postprocessing    
    for p in preproc:
        cutout = p(cutout)
    
    # segmentation
    cutout = method(cutout)
    
    cutout = cutout.astype(np.uint8)
    # apply preprocessing
    for p in postproc:
        cutout = p(cutout)
    
    # put the cutout back into a mask of the full image size
    h_cutout, w_cutout = cutout.shape[:2]
    if h_cutout != h_orig or w_cutout != w_orig:
        # cutout = cutout.astype(np.uint8)
        cutout = cv2.resize(cutout, (w_orig, h_orig), interpolation=cv2.INTER_AREA)
        cutout = cutout.astype(bool)

    mask = np.zeros(img.shape[:2], dtype=bool)
    mask[y1:y2, x1:x2] = cutout
    return mask


def apply_segmenation_to_match(img: np.ndarray, match: List[Union[Reflection, DirectVehicle]], method: SegmentationMethod, preproc: List[BasicImgOp], postproc: List[BasicImgOp]):
    
    masks = []
    for instance in match:
        mask = apply_segmentation(img, instance.pos, method, preproc, postproc)
        masks.append(mask)
    
    return masks


def calculate_match_iou(img: np.ndarray, match: List[Union[Reflection, DirectVehicle]], method: SegmentationMethod, preproc: List[BasicImgOp], postproc: List[BasicImgOp]):
    masks = apply_segmenation_to_match(img, match, method, preproc, postproc)
    iou = jaccard(masks)
    return iou


def visualize_annotator_masks(img: np.ndarray, masks: List[np.ndarray], annot_ids: List[int]):
    assert len(masks) == len(annot_ids)
    annot_imgs = []
    for mask, id in zip(masks, annot_ids):
        annot_img = img.copy()
        color = COLORS[id]
        color_matrix = np.zeros_like(annot_img)
        color_matrix = color
        mask = np.stack([mask]*3, axis=2)
        color_matrix *= mask
        color_matrix = color_matrix.astype(np.uint8)
        annot_img *= (mask == False)
        annot_img += color_matrix
        annot_imgs.append(annot_img)
    
    return annot_imgs
