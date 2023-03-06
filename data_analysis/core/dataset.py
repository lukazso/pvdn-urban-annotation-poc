import os
from re import A
from typing import Tuple
from tqdm import tqdm

from torch.utils.data import Dataset

from data_analysis.core.meta import *
from data_analysis.core.filter import *


class AI4ODDataset(Dataset):
    """

    Attributes
    ----------
        base_path (str): directory path to dataset

        img_subdir (str): directory inside a scene directory where 
            the images are stored. Default: 'images/left'

        annots_subdir (str): directory inside a scene directory
            where the annotations are stored. Default: 'annotations/left'

        load_images (bool): flag whether to load the images or not. 
            Default: True

        filters (List[BaseFilter]): list of filters to be applied on the    
            dataset. Default: []

        scenes (List[Scene]): List of Scene objects of the dataset.

        annotations (List[ImageAnnotation]): list of ImageAnnotation objects 
            of the dataset.

        img_infos (List[ImageInfo]): list of ImageInfo objects of the dataset.
        
    """
    def __init__(self, path: str, img_subdir: str = "images", load_images: bool = True, filters: List[BaseFilter] = None) -> None:
        super().__init__()

        if not os.path.isdir(path):
            raise NotADirectoryError(f"{path} is not a directory.")
        
        self.base_path = path
        self.load_images = load_images
        self.filters = filters if filters else []
        self.img_subdir = img_subdir

        self.scenes = []
        self.annotations = []
        self.img_infos = []

        for s in sorted(os.listdir(self.base_path)):
            scene_dir = os.path.join(self.base_path, s)
            if not os.path.isdir(scene_dir) or s.startswith("."):
                continue

            scene_id = int(s)

            scene = Scene.from_directory(
                src_dir=scene_dir, img_subdir=img_subdir, scene_id=scene_id
            )

            self.scenes.append(scene)
            self.img_infos += scene.img_infos
        
        for f in self.filters:
            self.scenes, self.annotations, self.img_infos = f(self.scenes, self.annotations, self.img_infos)
        
        # final paranoid checks
        n_in_scenes = 0
        for scene in self.scenes:
            n_in_scenes += len(scene.img_infos)
        
        assert n_in_scenes == len(self.img_infos), f"There is a mismatch between the number of images in Scene objects and in self.img_infos. This might be caused by a bugy filter."

        for scene in self.scenes:
            # read all segmentations
            pass


    def __repr__(self) -> str:
        return f"AI4ODDataset at {self.base_path}"

    def __len__(self):
        return len(self.img_infos)

    def __getitem__(self, index) -> Tuple:
        img_info = self.img_infos[index]

        img = img_info.get_img() if self.load_images else None
        
        return img, img_info
    
    def export(self, dst_dir: str):
        """This function exports your dataset. It might come handy when you 
            want to save your own version of the whole dataset (e.g., after 
            you applied some filters) for later usage.

        Args:
            dst_dir (str): Destination directory to store the dataset.
        """
        os.makedirs(dst_dir, exist_ok=True)
        
        for scene in tqdm(self.scenes, desc="Exporting scenes"):
            scene_name = str(scene.scene_id).zfill(5)
            scene_dir = os.path.join(dst_dir, scene_name)
            os.makedirs(scene_dir, exist_ok=True)
            scene.export(scene_dir)
            

if __name__ == "__main__":
    dataset = AI4ODDataset(path="/raid/datasets/ai4od/semseg-poc-paper/original-anonymized-filtered-types")
    a = 0
 