import os
import shutil


if __name__ == "__main__":
    data_dir = "/raid/datasets/ai4od/semseg-poc-paper/original-anonymized-filtered-types"
    assert os.path.exists(data_dir)

    scenes = os.listdir(data_dir)
    for scene in scenes:
        scene_dir = os.path.join(data_dir, scene)
        if os.path.isdir(scene_dir):
            mask_dir = os.path.join(scene_dir, "masks")
            mask_subdirs = os.listdir(mask_dir)
            for mask_subdir in mask_subdirs:
                if mask_subdir != "majority-best-params":
                    mask_subdir = os.path.join(mask_dir, mask_subdir)
                    print("Deleting", mask_subdir)
                    shutil.rmtree(mask_subdir)
