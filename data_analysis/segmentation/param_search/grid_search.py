import itertools
import pandas as pd
import numpy as np
import concurrent.futures as futures
from tqdm import tqdm
import os

from data_analysis.segmentation.utils import calculate_match_iou, apply_segmenation_to_match, jaccard
import data_analysis.segmentation.methods as methods
import data_analysis.segmentation.imgops as imgops
from skimage.morphology import disk

from data_analysis.core.dataset import AI4ODDataset
from data_analysis.core.filter import NumAnnotatorFilter
from data_analysis.boxes.match import get_matches
from data_analysis.boxes.utils import jaccard as jaccard_boxes
from data_analysis.segmentation.utils import jaccard as jaccard_segmentation
from data_analysis.core.meta import ReflectionType
from data_analysis.segmentation.methods import SegmentationMethod, DynamicThresholding


class GridSearchOptimizer:
    def __init__(self, search_space: dict, method: SegmentationMethod, dataset: AI4ODDataset, n_workers: int = os.cpu_count(), grayscale: bool = True, blur: bool = True, norm_minmax: bool = True) -> None:
        self.search_space = search_space
        self.combinations = self.make_combinations(self.search_space)
        self.dataset = dataset
        self.match_df = self.make_matches(self.dataset)
        self.method = method
        self.n_workers = n_workers

        self.grayscale = grayscale
        self.blur = blur
        self.norm_minmax = norm_minmax

    @staticmethod
    def make_matches(dataset):
        matches = {}
        img_infos = {}
        for img_info in dataset.img_infos:
            m, _ = get_matches(img_info, direct=False, indirect=True, distance_metric=jaccard_boxes, iou_thresh=0.01, exclusive=False)
            matches[img_info.img_name] = m
            img_infos[img_info.img_name] = img_info

        match_types = {t: [] for t in ReflectionType}
        match_types["ambiguous"] = []

        match_df = pd.DataFrame(columns=["img_id", "img_info", "match", "type", "box_iou"])
        for img_name, img_matches in matches.items():
            for match, iou in img_matches:
                types = [instance.type for instance in match]
                unique_types = list(set(types))

                if len(unique_types) > 1:
                    t = "ambiguous"
                else:
                    t = unique_types[0]
                
                tpl = (img_name, img_infos[img_name], match, t, iou)

                match_df.loc[len(match_df)] = tpl

        match_df["seg_iou"] = None
        match_df = match_df.sort_values(by=["img_id"])
        
        return match_df
    
    @staticmethod
    def make_combinations(search_space):
        keys = list(search_space.keys())
        keys.remove("method")
        values = [search_space[k] for k in keys]

        method_keys = list(search_space["method"].keys())
        method_values = list(search_space["method"].values())

        all_values = values + method_values
        
        combs = itertools.product(*all_values)
        combs_mapping = []
        for comb in combs:
            mapping = {keys[i]: comb[i] for i in range(len(keys))}
            method_params = comb[-len(method_keys):]
            method_mapping = {method_keys[i]: method_params[i] for i in range(len(method_keys))}
            mapping["method"] = method_mapping
            combs_mapping.append(mapping)
        
        return combs_mapping

    @staticmethod
    def fmin(params, df: pd.DataFrame, method: SegmentationMethod, grayscale: bool, blur: bool, norm_minmax: bool):
        preproc = []

        if grayscale:
            preproc.append(imgops.ToGrayscale())
        if blur:
            preproc.append(imgops.GaussianBlur((params["blur_window"], params["blur_window"]), params["blur_sigma"]))
        if norm_minmax:
            preproc.append(imgops.NormalizeMinMax())

        kernel_erode = disk(params["ksize_erode"])
        kernel_dilate = disk(params["ksize_dilate"])
        postproc = [
            imgops.Erosion(kernel_erode, params["iterations_erode"]),
            imgops.Dilation(kernel_dilate, params["iterations_dilate"]),
        ]

        method_params = params["method"]

        method = method(**method_params)

        prev_img_id = None
        
        seg_ious = []
        for _, row in df.iterrows():
            if row["img_id"] != prev_img_id:
                img = row["img_info"].get_img()
            masks = apply_segmenation_to_match(img, row["match"], method, preproc, postproc)
            seg_iou = jaccard_segmentation(masks)
            seg_ious.append(seg_iou)
        
        return seg_ious
    
    def _optimize_mp(self, df, pbar_desc: str = ""):
        seg_iou_results = []
        with futures.ProcessPoolExecutor(max_workers=self.n_workers) as exe:
            for result in tqdm(exe.map(GridSearchOptimizer.fmin, self.combinations, itertools.repeat(df), itertools.repeat(self.method), itertools.repeat(self.grayscale), itertools.repeat(self.blur), itertools.repeat(self.norm_minmax)), total=len(self.combinations), desc=pbar_desc):
                seg_iou_results.append(result)

        seg_iou_results = np.array(seg_iou_results)
        mean_seg_iou = list(seg_iou_results.mean(axis=1))

        mean_ious_sorted, combs_sorted = zip(*sorted(zip(mean_seg_iou, self.combinations), reverse=True, key=lambda x: x[0]))
        return mean_ious_sorted, combs_sorted       

    def _optimize_sp(self, df, pbar_desc: str = ""):
        seg_iou_results = []
        for combination in tqdm(self.combinations, desc=pbar_desc):
            result = self.fmin(combination, df, self.method, self.grayscale, self.blur, self.norm_minmax)

        seg_iou_results = np.array(seg_iou_results)
        mean_seg_iou = list(seg_iou_results.mean(axis=1))

        mean_ious_sorted, combs_sorted = zip(*sorted(zip(mean_seg_iou, self.combinations), reverse=True, key=lambda x: x[0]))
        return mean_ious_sorted, combs_sorted       
    
    def optimize_types_separately(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)

        col_names = ["type"]
        col_names += [k for k in self.search_space.keys() if k != "method"]
        col_names += [k for k in self.search_space["method"].keys()]
        col_names += ["mean_seg_iou", "mean_box_iou"]
        params_df = pd.DataFrame(columns=col_names)

        col_names.remove("type")
        for t in ReflectionType:
            df = self.match_df[self.match_df["type"] == t]
            mean_box_iou = df["box_iou"].mean()

            type_results_df = pd.DataFrame(columns=col_names)
            if len(df) > 0:
                mean_ious, combs = self._optimize_mp(df, pbar_desc=t.name)
                combs_values = []
                for comb in combs:
                    values = [v for k, v in comb.items() if k != "method"]
                    values += [v for v in comb["method"].values()]
                    combs_values.append(values)

                params_df.loc[len(params_df)] = [t.name] + combs_values[0] + [mean_ious[0], mean_box_iou]

                for mean_iou, params in zip(mean_ious, combs_values):
                    type_results_df.loc[len(type_results_df)] = params + [mean_iou, mean_box_iou]
                type_results_path = os.path.join(out_dir, f"{t.name}.csv")
                type_results_df.to_csv(type_results_path)
            
        best_params_path = os.path.join(out_dir, "best.csv")
        params_df.to_csv(best_params_path)
        print("Written results to", out_dir)


if __name__ == "__main__":
    from data_analysis.segmentation.methods import BMSThresholding
    from datetime import datetime

    search_space = {
        "ksize_erode": [1, 2],
        "iterations_erode": [0, 1],
        "ksize_dilate": [1, 2],
        "iterations_dilate": [1, 2],
        "method": {
            "n_thresholds": [5, 10],
            "t": [150, 175, 200, 225]
        }
    }
    # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M%S")  
    # this_dir = os.path.dirname(__file__)
    # out_dir = os.path.join(this_dir, "..", "out", "grid_search_bms", timestamp)

    data_dir = "/raid/datasets/ai4od/semseg-poc-paper/original/"

    filters = [NumAnnotatorFilter(2)]
    dataset = AI4ODDataset(data_dir, load_images=False, filters=filters)

    optim = GridSearchOptimizer(search_space, BMSThresholding, dataset, grayscale=False, blur=False, norm_minmax=False)

    num_combs = len(optim.combinations)
    num_workers = 1

    runtime_all = num_combs * 1.0
    runtime_per_worker = round(runtime_all / num_workers)
    print("Number of combinations:", num_combs)
    print(f"Expected runtime: {runtime_per_worker} min")

    optim.optimize_types_separately("")
