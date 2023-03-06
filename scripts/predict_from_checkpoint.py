import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd

from torchvision.prototype import datapoints, transforms as T
import torch
from torch.utils.data import DataLoader

from training.dataset import SemanticDataset
from training.train import LightningNetwork
from training.metrics import SemanticMetrics


COLOR = (255, 128, 0)

def make_overlay(img, mask):
    img = img.copy()
    img[mask] = COLOR
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--val-dir", "-v", type=str)
    parser.add_argument("--annot-subdir", "-a", type=str, default="masks/majority-best-params")
    parser.add_argument("--ckpt-path", "-c", type=str, help="path to checkpoint.ckpt")
    parser.add_argument("--out-dir", "-o", type=str)

    parser.add_argument("--scene-ids", nargs="+", type=int, default=None)
    
    parser.add_argument("--device", "-d", type=str, default="cuda")
    parser.add_argument("--batch-size", "-b", type=int, default=1)
    parser.add_argument("--num-workers", "-n", type=int, default=4)
    parser.add_argument("--conf-thresh", type=float, default=0.5)
    
    args = parser.parse_args()

    img_size = [240, 960]           # TODO: make parameter
    trans = T.Compose([
            T.Resize(img_size)      # (height, width)
        ])
    dataset = SemanticDataset(args.val_dir, annot_subdir=args.annot_subdir, img_subdir="images", transforms=trans)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.scene_ids is not None:
        included_imgs = []
        for scene_id in args.scene_ids:
            scene_id = str(scene_id).zfill(5)
            included_imgs += dataset.scene_id_mapping[scene_id]
        
        dataset.data = [data for data in dataset.data if data.id in included_imgs]

    os.makedirs(args.out_dir, exist_ok=True)
    out_dir_pred_mask = os.path.join(args.out_dir, "pred_masks")
    out_dir_pred_overlays = os.path.join(args.out_dir, "pred_overlays")
    out_dir_gt_mask = os.path.join(args.out_dir, "gt_masks")
    out_dir_gt_overlays = os.path.join(args.out_dir, "gt_overlays")
    out_dir_orig = os.path.join(args.out_dir, "orig")
    os.makedirs(out_dir_pred_mask, exist_ok=True)
    os.makedirs(out_dir_pred_overlays, exist_ok=True)
    os.makedirs(out_dir_gt_mask, exist_ok=True)
    os.makedirs(out_dir_gt_overlays, exist_ok=True)
    os.makedirs(out_dir_orig, exist_ok=True)

    model = LightningNetwork.load_from_checkpoint(args.ckpt_path)
    model = model.to(args.device)
    model.eval()

    normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    metrics = SemanticMetrics(num_classes=2).to(args.device)

    for i, (imgs, masks, img_ids) in tqdm(enumerate(dataloader)):
        imgs_orig = imgs.clone()
        imgs = imgs.to(args.device)
        masks = masks.to(args.device)
        imgs = normalize(imgs)
        
        pred_tensors = model(imgs)["out"]
        pred_probs = torch.softmax(pred_tensors, dim=1)
        pred_labels = pred_probs > args.conf_thresh
        metrics.update(pred_labels, masks)
        pred_labels = torch.argmax(pred_probs, dim=1)

        pred_labels = pred_labels.cpu().numpy()
        
        masks = masks.cpu().numpy()

        for img_orig, mask, pred_label, img_id in zip(imgs_orig, masks, pred_labels, img_ids):
            img = img_orig.permute(1, 2, 0).cpu().numpy()
            img *= 255
            img = img.astype(np.uint8)
            cv2.imwrite(os.path.join(out_dir_orig, img_id + ".png"), img)
            
            cv2.imwrite(os.path.join(out_dir_gt_mask, img_id + ".png"), (mask * 255).astype(np.uint8))
            gt_overlay = make_overlay(img, mask.astype(bool))
            cv2.imwrite(os.path.join(out_dir_gt_overlays, img_id + ".png"), gt_overlay)

            cv2.imwrite(os.path.join(out_dir_pred_mask, img_id + ".png"), (pred_label * 255).astype(np.uint8))
            
            pred_mask = pred_label.astype(bool)    # boolean mask

            img_overlay = make_overlay(img, pred_mask)
            cv2.imwrite(os.path.join(out_dir_pred_overlays, img_id + ".png"), img_overlay)
        
    iou = metrics.jaccard(average=False).cpu().numpy()
    dice = metrics.dice(average=False).cpu().numpy()
    acc = metrics.accuracy(average=False).cpu().numpy()

    df = pd.DataFrame(columns=["iou", "dice", "acc"], index=["reflection", "background", "avg"])
    df.loc["reflection", "iou"] = iou[1]
    df.loc["reflection", "dice"] = dice[1]
    df.loc["reflection", "acc"] = acc[1]
    df.loc["background", "iou"] = iou[0]
    df.loc["background", "dice"] = dice[0]
    df.loc["background", "acc"] = acc[0]
    df.loc["avg", "iou"] = np.mean(iou)
    df.loc["avg", "dice"] = np.mean(dice)
    df.loc["avg", "acc"] = np.mean(acc)

    print(df.to_markdown(tablefmt="github"))
