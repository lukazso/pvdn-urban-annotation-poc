import argparse
import os
import random

random.seed(123)

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def filter_num_annotators(data_dir, scenes, n_annots: int = 2):
    filtered_scenes = []
    for scene in scenes:
        scene_dir = os.path.join(data_dir, scene)
        annot_dirs = [d for d in os.listdir(scene_dir) if "annotations" in d]
        if len(annot_dirs) >= n_annots:
            filtered_scenes.append(scene)
    return filtered_scenes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--out-dir", type=str)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.0)

    args = parser.parse_args()

    scenes = sorted(os.listdir(args.data_dir))
    scenes = filter_num_annotators(args.data_dir, scenes, n_annots=2)

    random.shuffle(scenes)  # shuffle in place

    if args.test_size > 0:
        test_scenes = scenes[:int(len(scenes) * args.test_size)]
        scenes = scenes[int(len(scenes) * args.test_size):]

    scene_chunks = list(split(scenes, args.k))
    

    for i, chunk in enumerate(scene_chunks):
        train_path = os.path.join(args.out_dir, str(i), "train")
        val_path = os.path.join(args.out_dir, str(i), "val")
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(val_path, exist_ok=True)

        train_scenes = [s for c in scene_chunks for s in c if c != chunk]
        val_scenes = chunk

        for scene in train_scenes:
            src_dir = os.path.join(args.data_dir, scene)
            tgt_dir = os.path.join(train_path, scene)
            os.symlink(src_dir, tgt_dir)
        
        for scene in val_scenes:
            src_dir = os.path.join(args.data_dir, scene)
            tgt_dir = os.path.join(val_path, scene)
            os.symlink(src_dir, tgt_dir)
