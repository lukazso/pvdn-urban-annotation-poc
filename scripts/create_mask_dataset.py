import os
import json
from typing import List
import argparse
from tqdm import tqdm

from skimage.morphology import disk
import cv2
import numpy as np

from data_analysis.core.meta import ReflectionType
from data_analysis.segmentation.methods import SegmentationMethod, DynamicThresholding, AdaptiveGaussianThresholding, OtsuThresholding
from data_analysis.segmentation.aggregation import Aggregator, IntersectionAggregator, UnionAggregator, MajorityAggregator
from data_analysis.segmentation.utils import apply_segmentation
from data_analysis.segmentation.imgops import GaussianBlur, NormalizeMinMax, ToGrayscale, Erosion, Dilation
from data_analysis.segmentation.pipeline import SegmentationPipeline, MultiAnnotatorSegmentationPipeline

from data_analysis.core.dataset import AI4ODDataset
from data_analysis.core.filter import NumAnnotatorFilter

from data_analysis.boxes.match import get_matches


METHOD_LOOKUP = {
    "otsu": OtsuThresholding,
    "dynamic": DynamicThresholding,
    "adaptive_gaussian": AdaptiveGaussianThresholding
}

AGGREGATOR_LOOKUP = {
    "intersection": IntersectionAggregator,
    "union": UnionAggregator,
    "majority": MajorityAggregator
}


def create_mask_dataset(dataset: AI4ODDataset, mask_subdir: str, pipeline: MultiAnnotatorSegmentationPipeline):
    pbar = tqdm(dataset.scenes, total=len(dataset))
    for scene in pbar:
        # setup mask directory
        scene_dir = os.path.join(dataset.base_path, str(scene.scene_id).zfill(5))
        mask_dir = os.path.join(scene_dir, mask_subdir)
        os.makedirs(mask_dir, exist_ok=True)
        
        # for each image in the scene, find the bounding box matches
        for img_info in scene.img_infos:
            boxes = []
            types = []
            for annotator_id, annots in img_info.annots.items():
                b = [reflection.pos for reflection in annots.reflections]
                boxes.append(b)
                t = [reflection.type for reflection in annots.reflections]
                types.append(t)

            # apply the pipeline to the matches
            img = img_info.get_img()
            mask = pipeline.apply_to_multiple_annotators(img, boxes, types)
            
            img_id = img_info.img_name.split(".")[0]
            mask_path = os.path.join(mask_dir, f"{img_id}.png")
            mask = mask.astype(np.uint8) * 255
            cv2.imwrite(mask_path, mask)

            pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--param-path", type=str)
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--mask-subdir", type=str, default="masks/majority-best-params")
    parser.add_argument("--aggregation", type=str, default="majority", choices=("intersection", "union", "majority"))
    
    args = parser.parse_args()

    with open(args.param_path, "r") as f:
        param_collection = json.load(f)
    
    pipelines = {}
    for t, content in param_collection.items():
        t = ReflectionType[t]
        method = METHOD_LOOKUP[content["method"]]

        params = content["params"]
        method = method(**params["method_params"])

        preproc = [
            ToGrayscale(),
            GaussianBlur(w=[int(params["blur_window"])]*2, sigma=params["blur_sigma"]),
            NormalizeMinMax()
        ]
        postproc = [
            Erosion(kernel=disk(params["ksize_erode"]), iterations=params["iterations_erode"]),
            Dilation(kernel=disk(params["ksize_dilate"]), iterations=params["iterations_dilate"])
        ]
    
        
        pipeline = SegmentationPipeline(method, preproc, postproc)
        pipelines[t] = pipeline

    aggregator = AGGREGATOR_LOOKUP[args.aggregation]()
    multi_pipeline = MultiAnnotatorSegmentationPipeline(pipelines, aggregator)
    
    filters = [NumAnnotatorFilter(2)]
    dataset = AI4ODDataset(args.data_dir, load_images=False, filters=filters)
    
    create_mask_dataset(dataset, args.mask_subdir, multi_pipeline)