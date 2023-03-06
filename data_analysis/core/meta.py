from cmath import inf, pi
from datetime import datetime
from gettext import install
from random import shuffle
import shutil
import time
import os
import json
from typing import List, Union, Dict
import cv2
import numpy as np
from enum import Enum
from warnings import warn

import matplotlib.pyplot as plt


class ReflectionType(Enum):
    UNDEFINED = -1
    FLOOR = 0
    CAR = 1
    LINE = 2
    CURB = 3
    DIFFUSE = 4
    MIRROR = 5
    OTHER = 6


class AnnotatedObject:
    """ Class representing an annotated object.

    Attributes:
        pos (List): The position of the annotated object.
        direct (bool): Whether or not the annotated object is directly visible.
        id (int): The ID of the annotated object.
    """
    def __init__(self, pos: List, direct: bool, id: int) -> None:
        """Initializes an AnnotatedObject.

        Args:
            pos (List): The position of the annotated object.
            direct (bool): Whether or not the annotated object is directly visible.
            id (int): The ID of the annotated object.
        """
        self.pos = pos
        self.direct = direct
        self.id = id



class Reflection(AnnotatedObject):
    """Class representing a reflection.
    Attributes:
        type (ReflectionType): The type of the reflection.
    """
    def __init__(self, pos: List, type: ReflectionType, id: int) -> None:
        """Initializes a Reflection object.

        Args:
            pos (List): The position of the reflection.
            type (ReflectionType): The type of the reflection.
            id (int): The ID of the reflection.
        """
        super().__init__(pos, False, id)
        self.type = type

    def __repr__(self) -> str:
        """Returns a string representation of the Reflection object."""
        return f"Reflection (id: {self.id}, box: {self.pos})"

    @staticmethod
    def from_dict(annot: Dict):
        """Creates a Reflection object from a dictionary.

        Args:
            annot (Dict): The dictionary containing the reflection information.

        Returns:
            Reflection: The created Reflection object.
        """
        rtype = ReflectionType[annot["reflection_type"]]        
        inst = Reflection(
            pos=annot["bbox"],
            type=rtype,
            id=annot["iid"]
        )
        return inst


class Vehicle(AnnotatedObject):
    """Class representing a vehicle.

    Attributes:
        instances (List[Reflection] or None): List of reflections on the vehicle.
        pos (List): The position of the vehicle.
        direct (bool): Whether or not the vehicle is directly visible.
        id (int): The ID of the vehicle.

    """
    def __init__(self, pos: List, direct: bool, id: int, instances: Union[List[Reflection], None]) -> None:
        """Initializes a Vehicle object.

        Args:
            pos (List): The position of the vehicle.
            direct (bool): Whether or not the vehicle is directly visible.
            id (int): The ID of the vehicle.
            instances (List[Reflection] or None): List of reflections on the vehicle.
        """

        def __init__(self, pos, direct: bool, id: int, instances: Union[List[Reflection], None]) -> None:
            super().__init__(pos, direct, id)
            self.instances = instances if instances else []


class DirectVehicle(Vehicle):
    """ Contains the same attributes as Vehicle. Only difference is that the pos attribute is a list of 4 elements (x1, y1, x2, y2) and `direct` is always True.
    """
    def __init__(self, pos, id: int, instances: Union[List[Reflection], None]) -> None:
        assert isinstance(pos, list) or isinstance(pos, tuple)
        assert len(pos) == 4
        super().__init__(pos, True, id, instances)

    def __repr__(self) -> str:
        return f"Direct Vehicle (id: {self.id}, box: {self.pos})"

    @staticmethod
    def from_dict(annot: Dict):
        instances = []
        for inst_annot in annot["instances"]:
            if inst_annot["direct"] == False:
                inst = Reflection.from_dict(inst_annot)
                instances.append(inst)
        vehicle = DirectVehicle(
            pos=annot["bbox"], id=annot["oid"], instances=instances
        )
        return vehicle


class IndirectVehicle(Vehicle):
    """ Contains the same attributes as Vehicle. Only difference is that the pos attribute is a list of 2 elements (x, y) and `direct` is always False.
    """
    def __init__(self, pos, id: int, instances: Union[List[Reflection], None]) -> None:
        assert isinstance(pos, list) or isinstance(pos, tuple)
        assert len(pos) == 2
        super().__init__(pos, False, id, instances)

    def __repr__(self) -> str:
        return f"Indirect Vehicle (id: {self.id}, pos: {self.pos})"

    @staticmethod
    def from_dict(annot: Dict):
        instances = []
        for inst_annot in annot["instances"]:
            if inst_annot["direct"] == False:
                inst = Reflection.from_dict(inst_annot)
                instances.append(inst)
        vehicle = IndirectVehicle(
            pos=annot["pos"], id=annot["oid"], instances=instances
        )
        return vehicle


class ImageInfo:
    COLORS = {  # BGR
        "vehicle": (255, 128, 0),    # blue-ish
        "reflection": (0, 128, 255)   # orange
    }
    THICKNESS = 2
    RADIUS = 3
    def __init__(self, img_name: str, img_dir: str, scene=None, annots={}) -> None:
        self.img_dir = img_dir
        self.img_name = img_name
        self.scene = scene
        self.annots = annots

    @property
    def img_path(self):
        return os.path.join(self.img_dir, self.img_name)
    
    def get_img(self) -> np.array:
        """
        Returns:
            np.array: image as np.uint8 of shape [h, w, c] or [h, w] if     
                grayscale.
        """
        return cv2.imread(self.img_path)

    def __repr__(self) -> str:
        return f"ImageInfo {self.img_name}"

    def __eq__(self, o) -> bool:
        return self.img_name == o.img_name
    
    def visualize(self, annotators: Union[List[int], None] = None, vehicles: bool = True, reflections: bool = True):
        img = self.get_img()
        if not annotators:
            annotators = list(self.annots.keys())

        for annotator in sorted(annotators):
            annot = self.annots[annotator]
            for vehicle in annot.vehicles:
                if vehicles:
                    if isinstance(vehicle, IndirectVehicle):
                        center = vehicle.pos
                        img = cv2.circle(img, center, self.RADIUS, self.COLORS["vehicle"], cv2.LINE_AA)
                    else:
                        x1, y1, x2, y2 = vehicle.pos
                        img = cv2.rectangle(img, (x1, y1), (x2, y2), self.COLORS["vehicle"], self.THICKNESS, cv2.LINE_AA)
                if reflections:
                    for instance in vehicle.instances:
                        x1, y1, x2, y2 = instance.pos
                        img = cv2.rectangle(img, (x1, y1), (x2, y2), self.COLORS["reflection"], self.THICKNESS, cv2.LINE_AA)
        
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


class ImageAnnotation:
    def __init__(self, annotator: str, vehicles: List[Vehicle]) -> None:
        self.annotator = annotator
        self.vehicles = vehicles

    @property
    def reflections(self):
        insts = [inst for vehicle in self.vehicles for inst in vehicle.instances]
        return insts

    def __repr__(self) -> str:
        return f"Annotation from {self.annotator}"

    @staticmethod
    def from_dict(annot: Dict):
        annotator = annot["annotator"]
        vehicle_annots = annot["objects"]
        vehicles = []
        for vehicle_annot in vehicle_annots:
            if vehicle_annot["direct"]:
                vehicle = DirectVehicle.from_dict(vehicle_annot)
            else:
                vehicle = IndirectVehicle.from_dict(vehicle_annot)
            vehicles.append(vehicle)

        annotation = ImageAnnotation(
            annotator=annotator, vehicles=vehicles
        )
        return annotation

    @staticmethod
    def from_json(path: str):
        assert path.endswith(".json"), f"{path} is not a .json file."
        assert os.path.exists(path), f"{path} does not exist."
        with open(path, "r") as f:
            d = json.load(f)

        if d is None:
            return None

        annot = ImageAnnotation.from_dict(d)
        return annot


class Scene:
    def __init__(self, img_infos: List[ImageInfo], scene_id: int) -> None:
        self.img_infos = img_infos
        self.scene_id = scene_id

        for img_info in self.img_infos:
            img_info.scene = self
    
    def __repr__(self) -> str:
        return f"Scene {self.scene_id}"

    @staticmethod
    def from_directory(src_dir: str, img_subdir: str = f"images", scene_id: int = None):
        assert os.path.exists(src_dir), f"{src_dir} does not exist."

        scene_id = scene_id if scene_id else \
            int(src_dir.split(os.sep)[-1])

        # load image names
        img_dir = os.path.join(src_dir, img_subdir)
        imgs = sorted(os.listdir(img_dir))

        # load annotations
        annot_dirs = sorted([os.path.join(src_dir, a) for a in os.listdir(src_dir) if a.startswith("annotations")])
        annot_ids = [int(a.split(os.sep)[-1].replace("annotations", "")) for a in annot_dirs]
        annot_dir_exists = True
        if len(annot_dirs) == 0:
            warn(f"Scene {scene_id} does not contain any annotations. This scene will be initialized without annotations.")
            annot_dir_exists = False
        img_infos = []
            
        # load annotation for each image
        scene_annots = {a: [] for a in annot_ids}
        for img in imgs:
            img_id = img.split(".")[0]

            img_annots = {a: None for a in annot_ids}

            if annot_dir_exists:
                for id, annot_dir in zip(annot_ids, annot_dirs):
                    annot_file = img_id + ".json"
                    annot_path = os.path.join(annot_dir, annot_file)
                    if os.path.exists(annot_path):
                        annot = ImageAnnotation.from_json(annot_path)
                    img_annots[id] = annot
                    scene_annots[id].append(annot)
            
                img_info = ImageInfo(img_name=img, img_dir=img_dir, scene=None, annots=img_annots)
                img_infos.append(img_info)

        scene = Scene(img_infos=img_infos, scene_id=scene_id)
        return scene
