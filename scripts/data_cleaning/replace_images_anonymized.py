import os
import shutil
from tqdm import tqdm


def get_anonymized_imgs(src_dir):
    img_paths = {}
    for img_name in os.listdir(src_dir):
        img_paths[img_name] = os.path.join(src_dir, img_name)
    return img_paths


def get_dst_imgs(dst_dir):
    img_paths = {}
    scenes = os.listdir(dst_dir)
    for scene in scenes:
        scene_dir = os.path.join(dst_dir, scene)
        if os.path.isdir(scene_dir):
            img_dir = os.path.join(scene_dir, "images")
            if os.path.exists(img_dir):
                for img_name in os.listdir(img_dir):
                    img_paths[img_name] = os.path.join(img_dir, img_name)
    return img_paths


if __name__ == "__main__":
    src_dir = "/raid/datasets/ai4od/semseg-poc-paper/images-anonymized"
    assert os.path.exists(src_dir)

    dst_dir = "/raid/datasets/ai4od/semseg-poc-paper/original-anonymized-filtered-types"

    src_imgs = get_anonymized_imgs(src_dir)
    dst_imgs = get_dst_imgs(dst_dir)

    pbar = tqdm(src_imgs.items())
    for img_name, src_path in pbar:
        if img_name in dst_imgs.keys():
            dst_path = dst_imgs[img_name]
            shutil.copy(src_path, dst_path)
            pbar.desc = f"Replaced {dst_path} with {src_path}"
        else:
            pbar.desc = f"Could not find {img_name} in {dst_dir}"
        pbar.update(1)
