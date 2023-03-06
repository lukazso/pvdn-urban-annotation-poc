import json
import os
from tqdm import tqdm

if __name__ == "__main__":
    data_dir = "/raid/datasets/ai4od/semseg-poc-paper/original-anonymized-filtered-types"
    assert os.path.exists(data_dir)

    scenes = sorted(os.listdir(data_dir))
    for scene in tqdm(scenes):
        scene_dir = os.path.join(data_dir, scene)
        if os.path.isdir(scene_dir):
            annot_dirs = [d for d in os.listdir(scene_dir) if "annotations" in d]
            for annot_dir in annot_dirs:
                annot_dir = os.path.join(scene_dir, annot_dir)
                annot_files = os.listdir(annot_dir)
                for annot_file in annot_files:
                    annot_path = os.path.join(annot_dir, annot_file)
                    with open(annot_path, "r") as f:
                        data = json.load(f)
                    
                    data.pop("date", None)
                    data.pop("task_id", None)
                    for obj in data["objects"]:
                        obj.pop("img_meta", None)
                        obj.pop("date", None)
                        for inst in obj["instances"]:
                            inst.pop("date", None)
                            inst.pop("img_meta", None)
                            inst.pop("pos", None)
                            inst.pop("rear", None)

                    # save the changes
                    with open(annot_path, "w") as f:
                        json.dump(data, f, indent=4, sort_keys=True)
