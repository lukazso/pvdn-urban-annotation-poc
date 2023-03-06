import os
from datetime import datetime
import argparse

from data_analysis.core.dataset import AI4ODDataset
from data_analysis.core.filter import NumAnnotatorFilter
from data_analysis.segmentation.methods import DynamicThresholding

from data_analysis.segmentation.param_search.grid_search import GridSearchOptimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", "-d", type=str, required=True)
    parser.add_argument("--out-dir", "-o", type=str, default=None)
    args = parser.parse_args()

    search_space = {
        "blur_window": [5, 9],
        "blur_sigma": [1.0, 2.0],
        "ksize_erode": [1, 3],
        "iterations_erode": [0, 1],
        "ksize_dilate": [1, 3],
        "iterations_dilate": [0, 1],
        "method": {
            "k": [0.01, 0.03, 0.05, 0.07, 0.09],
            "w": [10, 30, 50, 70]
        }
    }
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M%S")  
    this_dir = os.path.dirname(__file__)

    out_dir = args.out_dir
    if args.out_dir is None:
        out_dir = os.path.join(this_dir, "..", "..", "out", "grid_search", timestamp)
    
    data_dir = args.data_dir

    filters = [NumAnnotatorFilter(2)]
    dataset = AI4ODDataset(data_dir, load_images=False, filters=filters)

    optim = GridSearchOptimizer(search_space, DynamicThresholding, dataset)
    optim.optimize_types_separately(out_dir)
