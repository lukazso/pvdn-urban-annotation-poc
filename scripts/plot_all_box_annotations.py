import os
import cv2
from tqdm import tqdm

from data_analysis.core.dataset import AI4ODDataset
from data_analysis.core.filter import NumAnnotatorFilter
from data_analysis.core.utils import visualize_different_annotators

data_dir = "/raid/datasets/ai4od/semseg-poc-paper/original/"
assert os.path.isdir(data_dir), "Your data directory is not a directory."

filters = [NumAnnotatorFilter(2)]
dataset = AI4ODDataset(data_dir, load_images=False, filters=filters)

for scene in tqdm(dataset.scenes):
    scene_dir = os.path.join(data_dir, str(scene.scene_id).zfill(5))
    out_dir = os.path.join(scene_dir, "visualization", "box_annotations")
    os.makedirs(out_dir, exist_ok=True)
    for img_info in scene.img_infos:
        img_viz, _ = visualize_different_annotators(img_info)
        out_path = os.path.join(out_dir, img_info.img_name)
        cv2.imwrite(out_path, img_viz)
