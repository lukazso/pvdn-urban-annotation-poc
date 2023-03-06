import json
import os
from data_analysis.core.meta import ReflectionType
from tqdm import tqdm


if __name__ == "__main__":
    data_dir = "/raid/datasets/ai4od/semseg-poc-paper/original-anonymized-filtered-types"
    assert os.path.exists(data_dir)

    relevant_types = [
        ReflectionType.LINE.name,
        ReflectionType.DIFFUSE.name,
        ReflectionType.MIRROR.name
    ]

    scenes = os.listdir(data_dir)
    for scene in tqdm(scenes):
        scene_dir = os.path.join(data_dir, scene)
        annot_dirs = [d for d in os.listdir(scene_dir) if "annotations" in d]
        for annot_dir in annot_dirs:
            annot_dir = os.path.join(scene_dir, annot_dir)
            annots = os.listdir(annot_dir)
            for annot in annots:
                annot_path = os.path.join(annot_dir, annot)
                with open(annot_path, "r") as f:
                    data = json.load(f)
                for obj in data["objects"]:
                    for inst in obj["instances"]:
                        if inst["reflection_type"] in relevant_types:
                            inst["reflection_type"] = ReflectionType.OTHER.name
                with open(annot_path, "w") as f:
                    json.dump(data, f)
