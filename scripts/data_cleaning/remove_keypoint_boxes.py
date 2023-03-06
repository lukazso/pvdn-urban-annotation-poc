import os
import shutil
from tqdm import tqdm
import json


if __name__ == "__main__":
    data_dir = "/raid/datasets/ai4od/semseg-poc-paper/original-anonymized-filtered-types"
    assert os.path.exists(data_dir)

    scenes = os.listdir(data_dir)
    for scene in tqdm(scenes):
        scene_dir = os.path.join(data_dir, scene)
        if os.path.isdir(scene_dir):
            annot_dirs = [d for d in os.listdir(scene_dir) if "annotations" in d]
            for annot_dir in annot_dirs:
                annot_dir = os.path.join(scene_dir, annot_dir)
                annots = os.listdir(annot_dir)
                for annot in annots:
                    annot_path = os.path.join(annot_dir, annot)
                    with open(annot_path, "r") as f:
                        data = json.load(f)
                    filtered_objects = []
                    for obj in data["objects"]:
                        # print(obj["direct"])
                        if obj["direct"]:
                            x1, y1, x2, y2 = obj["bbox"]
                            height = y2 - y1
                            width = x2 - x1
                            if height <= 1 or width <= 1:
                                print("Object:", data["img_name"], obj["bbox"], "Num instances:", len(obj["instances"]))
                                num_instances = len(obj["instances"])
                                if num_instances == 0:
                                    continue
                                else:
                                    obj = {k: v for k, v in obj.items() if k != "bbox"}
                                    obj["pos"] = [x1, y1]
                                    obj["direct"] = False

                        filtered_instances = []
                        for inst in obj["instances"]:
                            x1, y1, x2, y2 = inst["bbox"]
                            height = y2 - y1
                            width = x2 - x1
                            if height <= 1 or width <= 1:
                                print("Instance:", data["img_name"], inst["bbox"])
                            else:
                                filtered_instances.append(inst)
                                
                        if len(filtered_instances) > 0:
                            obj["instances"] = filtered_instances
                            filtered_objects.append(obj)
                            
                    with open(annot_path, "w") as f:
                        data["objects"] = filtered_objects
                        json.dump(data, f, indent=4, sort_keys=True)

