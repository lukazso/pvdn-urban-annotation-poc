"""

Optimization follows this logic:

1. Construct all the relevant bounding box matches PER IMAGE, which will be taken for optimization. The matches will be stored in a dictionary with image names as keys and matches as values. A match is a list of Reflection objects. Further, all ImageInfo objects will be stored in a dictionary with image names as keys and the corresponding ImageInfo object as values.

2. For each match, calculate the IoU.

3. Calculate the fitness per match. The fitness function will be the sum of the IoU and the average size of the segmentation w.r.t. its bounding box.

4. Average the fitness over all matches.

"""

from typing import List, Tuple, Dict
from statistics import mean
import os
import json

import numpy as np
from skimage.morphology import disk
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK

import data_analysis.segmentation.imgops as imgops
import data_analysis.segmentation.methods as methods
from data_analysis.segmentation.utils import apply_segmentation, jaccard
from data_analysis.boxes.match import get_matches
from data_analysis.core.meta import ImageInfo, Reflection


def get_area_in_box(masks: List[np.ndarray], boxes: List[Tuple[int, int, int, int]]):
    ratios = []
    for mask, box in zip(masks, boxes):
        x1, y1, x2, y2 = box
        cutout = mask[y1:y2, x1:x2]
        total_area = (x2 - x1) * (y2 - y1)
        masked_area = np.count_nonzero(cutout)
        ratio = masked_area / total_area
        ratios.append(ratio)
    return ratios


class NpEncoder(json.JSONEncoder):
    """ Source: https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class SegmentationPipeline:
    def __init__(self, k, w, blur_window, blur_sigma, ksize_erode, ksize_dilate, iterations_erode, iterations_dilate):
        self.preproc = [
            imgops.ToGrayscale(),
            imgops.GaussianBlur((blur_window, blur_window), blur_sigma),
            imgops.NormalizeMinMax(),
        ]

        kernel_erode = disk(ksize_erode)
        kernel_dilate = disk(ksize_dilate)
        self.postproc = [
            imgops.Erosion(kernel_erode, iterations_erode),
            imgops.Dilation(kernel_dilate, iterations_dilate),
        ]

        self.method = methods.DynamicThresholding(k, w)
    
    def __call__(self, img, boxes):
        masks = []
        for box in boxes:
            mask = apply_segmentation(img, box, self.method, self.preproc, self.postproc)
            masks.append(mask)
        
        iou = jaccard(masks)
        return iou, masks


class Optimizer:
    _AREA_LOSS_CHOICES = ["min", "max", None]
    def __init__(self, search_space: List, img_infos: Dict[str, ImageInfo], matches: Dict[str, List[List[Reflection]]], max_evals: int, out_dir: str, optim_algo=tpe.suggest, area_loss: str = None) -> None:
        self.search_space = search_space

        self.optim_algo = optim_algo
        self.max_evals = max_evals

        if not area_loss in self._AREA_LOSS_CHOICES:
            raise ValueError(f"area_loss must be one of {self._AREA_LOSS_CHOICES}.")
        self.area_loss = area_loss

        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        
        self.matches = {}
        self.img_infos = img_infos
        self.matches = matches
        
        # for logging
        self.history = []
    
    def optimize(self):
        best_params = fmin(
            fn=self.objective_func,
            space=self.search_space,
            algo=self.optim_algo,
            max_evals=self.max_evals
        )
        best_params = space_eval(self.search_space, best_params)
        
        log = {
            "history": self.history,
            "best_params": best_params
        }
        log_path = os.path.join(self.out_dir, "log.json")
        with open(log_path, "w") as f:
            json.dump(log, f, cls=NpEncoder, indent=4)

        print("Best parameters:")
        print(json.dumps(best_params, indent=4, cls=NpEncoder))

        print("Written logs to", log_path)

    def objective_func(self, params):
        # construct the segmentation pipeline for the current parameter set
        pipeline = SegmentationPipeline(**params)

        print(params)

        # for each match, calculate the iou and mask size relative to the bounding box size
        seg_ious = []
        avg_area_ratios = []
        for matches, img_info in zip(self.matches.values(), self.img_infos.values()):
            if len(matches) > 0:
                img = img_info.get_img()
                
                for match in matches:
                    boxes = [instance.pos for instance in match]
                    try:
                        seg_iou, masks = pipeline(img, boxes)
                        area_ratios = get_area_in_box(masks, boxes)
                    except ZeroDivisionError:
                        seg_iou = 0.
                        area_ratios = [0.] * len(boxes)
                        
                    avg_area_ratios.append(mean(area_ratios))
                    seg_ious.append(seg_iou)
        
        iou = mean(seg_ious)
        area_ratio = mean(avg_area_ratios)
        # loss = (1 - iou) + (1 - area_ratio)
        iou_loss = 1 - iou
        
        # logging
        params["iou_loss"] = iou_loss

        area_loss = 0.0
        if self.area_loss == "min":
            area_loss = area_ratio
        elif self.area_loss == "max":
            area_loss = 1 - area_ratio
        
        params["area_loss"] = area_loss
        self.history.append(params)

        loss = area_loss + iou_loss
        return loss


if __name__ == "__main__":
    from data_analysis.core.dataset import AI4ODDataset
    from data_analysis.core.filter import NumAnnotatorFilter
    from data_analysis.boxes.match import get_matches
    from data_analysis.boxes.utils import jaccard as jaccard_boxes
    from data_analysis.core.meta import ReflectionType
    from datetime import datetime
    import argparse

    this_dir = os.path.dirname(__file__)
    out_dir = os.path.join(this_dir, "..", "..", "out", "optimization", "segmentation", "types")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", "-d", type=str, default="/raid/datasets/ai4od/semseg-poc-paper/original/")
    parser.add_argument("--out-dir", "-o", type=str, default=out_dir)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--max-evals", type=int, default=100)
    parser.add_argument("--area-loss", type=str, default=None, choices=("min", "max"))

    args = parser.parse_args()
    data_dir = args.data_dir
    
    name = args.name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M%S")    
    if name:
        out_dir = os.path.join(args.out_dir, timestamp + "_" + name)
    else:
        out_dir = os.path.join(args.out_dir, timestamp)
    
    max_evals = args.max_evals
    area_loss = args.area_loss

    filters = [NumAnnotatorFilter(2)]
    dataset = AI4ODDataset(data_dir, load_images=False, filters=filters)

    matches = {}
    img_infos = {}
    for img_info in dataset.img_infos:
        m, _ = get_matches(img_info, direct=False, indirect=True, distance_metric=jaccard_boxes, iou_thresh=0.01, exclusive=False)
        matches[img_info.img_name] = m
        img_infos[img_info.img_name] = img_info

    match_types = {t: [] for t in ReflectionType}
    match_types["ambiguous"] = []

    for img_name, img_matches in matches.items():
        for match, _ in img_matches:
            types = [instance.type for instance in match]
            unique_types = list(set(types))

            tpl = (img_name, match)
            if len(unique_types) > 1:
                match_types["ambiguous"].append(tpl)
            else:
                match_types[unique_types[0]].append(tpl)

    for key, values in match_types.items():
        print(f"{key}: {len(values)}")

    search_space = {
            "k": hp.quniform("k", 0.01, 0.1, q=0.01),
            "w": hp.choice("w", np.arange(10, 100, step=10, dtype=int)),
            "blur_window": hp.choice("blur_window", np.arange(3, 17, step=2, dtype=int)),
            "blur_sigma": hp.quniform("blur_sigma", 0.0, 5.0, q=0.5),
            "ksize_erode": hp.choice("ksize_erode", np.arange(0, 3, dtype=int)),
            "ksize_dilate": hp.choice("ksize_dilate", np.arange(0, 3, dtype=int)),
            "iterations_erode": hp.choice("iterations_erode", np.arange(0, 3, dtype=int)),
            "iterations_dilate": hp.choice("iterations_dilate", np.arange(0, 3, dtype=int))
        }

    for t, matches in match_types.items():
        if t == "ambiguous" or len(matches) == 0:
            continue
        
        m = {k: [] for k, _ in matches}
        for k, v in matches:
            m[k].append(v)

        _out_dir = os.path.join(out_dir, t.name)
        print("Optimizing", t.name)
        print("Number of matches:", len(m))
        optim = Optimizer(search_space, img_infos, m, max_evals, _out_dir, area_loss=area_loss)
        optim.optimize()