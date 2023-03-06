import os
from typing import Tuple
from tqdm import tqdm
from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from torchvision.prototype import datapoints, transforms as T
from torchvision.prototype.transforms import functional as F

from training.filters import BaseFilter

import cv2
import numpy as np


def calculate_class_weights(dataloader: DataLoader, num_classes: int, show_pbar: bool = True, sync: bool = True):
    class_count = [0] * num_classes
    
    total = len(dataloader.dataset)
    bs = dataloader.batch_size
    pbar = tqdm(dataloader, total=total, disable=not show_pbar, desc="Calculating class weights by evaluating training images.")
    for _, masks, _ in pbar:
        uniques = torch.unique(masks)
        for i in uniques:
            count = torch.count_nonzero(masks == i)
            class_count[i] += count
        pbar.update(n=bs)   
    
    class_count = torch.Tensor(class_count)

    if sync:
        tensor_list = [torch.zeros_like(class_count) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list, class_count)
        class_count = torch.stack(tensor_list, dim=0).sum(dim=0)

    class_weights = class_count.min() / class_count
    return class_weights
        

class ImageContainer:
    def __init__(self, scene: str, id: str, img_path: str, mask_path: str) -> None:
        self.scene: str = scene
        self.id: str = id
        self.img_path: str = img_path
        self.mask_path: str = mask_path
    
    def get_mask(self):
        mask = cv2.imread(self.mask_path, 0)
        mask[mask > 0] = 1
        return mask
    
    def get_img(self):
        img = cv2.imread(self.img_path)
        return img

    def load(self):
        img = self.get_img()
        mask = self.get_mask()
        return img, mask

    def __repr__(self):
        return self.id

class SemanticDataset(Dataset):
    def __init__(self, path: str, img_subdir: str, annot_subdir: str = "masks/majority-best-params/", transforms = None, filters: List[BaseFilter] = None) -> None:
        super().__init__()
        
        assert os.path.isdir(path)
        
        self.path = path
        self.img_subdir = img_subdir
        self.annot_subdir = annot_subdir

        self.transforms = transforms
        
        self.data = []
        self.scenes = sorted(os.listdir(self.path))

        self.filters = filters if filters else []

        self.scene_id_mapping = {}
        for scene in self.scenes:
            scene_dir = os.path.join(self.path, scene)
            if not os.path.isdir(scene_dir):
                continue
            
            img_dir = os.path.join(scene_dir, self.img_subdir)
            mask_dir = os.path.join(scene_dir, self.annot_subdir)
            if not os.path.isdir(mask_dir) or not os.path.isdir(img_dir):
                continue

            img_names = os.listdir(img_dir)
            mask_names = os.listdir(mask_dir)
            
            self.scene_id_mapping[scene] = []

            for i, mask_name in enumerate(sorted(mask_names)):
                
                img_id = mask_name.split(".")[0]
                img_name = img_id + ".jpg"

                img_path = os.path.join(img_dir, img_name)
                mask_path = os.path.join(mask_dir, mask_name)
                
                assert os.path.isfile(img_path)
                assert os.path.isfile(mask_path)
                
                container = ImageContainer(scene, img_id, img_path, mask_path)
                self.data.append(container)
                self.scene_id_mapping[scene].append(img_id)
        
        for filter in self.filters:
            self.data = filter(self.data)        
    
        self.to_img_tensor = T.Compose(
            [
                T.ToImageTensor(),
                T.ConvertImageDtype(torch.float32)
            ]
        )
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        img, mask = data.load()
        img_id = data.id

        # crop img & mask
        # TODO: make this parameterizable
        img = img[230:450, :, :]
        mask = mask[230:450, :]

        mask_features = datapoints.Mask(mask)
        img_features = self.to_img_tensor(img)
        
        if self.transforms:
            img_features, mask_features = self.transforms(img_features, mask_features)
        
        return img_features.data, mask_features.data, img_id


class SemanticDataModule(pl.LightningDataModule):
    def __init__(self, train_dir: str, val_dir: str, test_dir: str, batch_size: int, num_workers: int, img_size: Tuple[int, int], img_subdir: str = "images/", mask_subdir: str = "masks/majority-best-params/", filters: List[BaseFilter] = None) -> None:
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir

        self.img_subdir = img_subdir
        self.mask_subdir = mask_subdir

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.img_size = img_size    # (height, width)

        self.filters = filters
        
    def train_dataloader(self):
        # TODO: add transform
        trans = T.Compose([
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            T.Resize(self.img_size)    # (height, width)
        ])
        dataset = SemanticDataset(self.train_dir, self.img_subdir, self.mask_subdir, transforms=trans, filters=self.filters)
        dataloader = DataLoader(
            dataset=dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=True
        )
        return dataloader
    
    def val_dataloader(self):
        # TODO: add transform
        trans = T.Compose([
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            T.Resize(self.img_size)    # (height, width)
        ])
        dataset = SemanticDataset(self.val_dir, self.img_subdir, self.mask_subdir, transforms=trans)
        dataloader = DataLoader(
            dataset=dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False
        )
        return dataloader
