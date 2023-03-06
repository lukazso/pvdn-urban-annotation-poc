import os
from typing import List
import wandb
from datetime import datetime

import torch
import numpy as np
import random

import torch.nn as nn
import torch.distributed as dist
from torchvision.models import ResNet50_Weights, ResNet101_Weights, MobileNet_V3_Large_Weights
from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import segmentation_models_pytorch as smp

from training.dataset import SemanticDataModule, calculate_class_weights
from training.loss import FocalLoss2d
from training.metrics import SemanticMetrics
from training.filters import EmptyFilter


BACKBONE_LOOKUP = {
    "resnet50": (deeplabv3_resnet50, ResNet50_Weights),
    "resnet101": (deeplabv3_resnet101, ResNet101_Weights),
    "mobilenet_v3_large": (deeplabv3_mobilenet_v3_large, MobileNet_V3_Large_Weights)
}


class LightningNetwork(pl.LightningModule):
    def __init__(self, model: nn.Module, num_classes: int, lr: float, wd: float, betas: List = [0.9, 0.999], gamma: float = 4.,  conf_thres: float = 0.5, class_weights: torch.Tensor = None, tmax: int = 50, eta_min: float = 0.00001, iou_weight: float = 0.5, focal_weight: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.lr = lr
        self.wd = wd
        self.betas = betas
        self.gamma = gamma
        self.class_weights = class_weights

        self.conf_thresh = conf_thres

        # lr scheduler params
        self.tmax = tmax
        self.eta_min = eta_min

        self.model = model
        self.aux_loss = self.model.aux_classifier is not None

        self.save_hyperparameters()

        # self.focal_loss = FocalLoss2d(gamma=gamma, weight=class_weights)
        self.losses = {
            "iou": (iou_weight, smp.losses.JaccardLoss(mode="multiclass", from_logits=True)),
            "focal": (focal_weight, smp.losses.FocalLoss(mode="multiclass", gamma=gamma, alpha=class_weights, reduction="mean"))
        }
        # self.focal_weight = focal_weight
        # self.iou_weight = iou_weight
        # self.focal_loss = smp.losses.FocalLoss(mode="multilabel", gamma=gamma, alpha=class_weights, reduction="mean")
        # self.iou_loss = smp.losses.JaccardLoss(mode="multilabel", reduction="mean", classes=[1], from_logits=True)

        self.val_metrics = SemanticMetrics(num_classes=num_classes)
        # self.test_metrics = SemanticMetrics(num_classes=num_classes)

        self.train_loss_values = {k: [] for k in self.losses}
        self.train_loss_values["aux"] = []
        self.train_loss_values["total"] = []

        self.val_loss_values = {k: [] for k in self.losses}
        self.val_loss_values["aux"] = []
        self.val_loss_values["total"] = []
        # self.train_losses = {
        #     "focal": [],
        #     "iou": [],
        #     "aux": [],
        # }
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd, betas=self.betas,)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.tmax, self.eta_min
        )
        metric = "avg_loss/val"

        return [optimizer], [{"scheduler": scheduler, "metric": metric}]      
    
    def training_step(self, batch, batch_idx):
        imgs, masks, ids = batch
        masks = masks.long()
        out = self.model(imgs)
        pred_logits = out["out"]

        total_loss = 0
        for key, (weight, fn) in self.losses.items():
            loss = fn(pred_logits, masks)
            total_loss += weight * loss
            self.train_loss_values[key].append(loss.clone().detach())
        if self.aux_loss:
            aux_loss = out["aux"]
            aux_loss = torch.mean(aux_loss)
            total_loss = aux_loss + total_loss
            self.train_loss_values["aux"].append(aux_loss.clone().detach())
        
        self.train_loss_values["total"].append(total_loss.clone().detach())

        # if batch_idx % self.log_every == 0:
        #     self.log("step_loss/train/focal", focal_loss, rank_zero_only=True)
        #     if self.aux_loss:
        #         self.log("step_loss/train/aux", aux_loss, rank_zero_only=True)
        assert not total_loss.isnan().any()
        assert not total_loss.isinf().any()

        return total_loss
    
    def training_epoch_end(self, losses):
        if isinstance(losses[0], dict):
            losses = [o["loss"] for o in losses]

        for key, values in self.train_loss_values.items():
            tensor = torch.Tensor(values).to(self.device)
            self.sync_and_log_losses_on_epoch(tensor, f"avg_loss/train/{key}")

        self.train_loss_values = {k: [] for k in self.train_loss_values.keys()}
    
    def validation_step(self, batch, batch_idx):
        imgs, masks, ids = batch
        masks = masks.long()
        out = self.model(imgs)
        pred_logits = out["out"]

        total_loss = 0
        for key, (weight, fn) in self.losses.items():
            loss = fn(pred_logits, masks)
            total_loss += weight * loss
            self.val_loss_values[key].append(loss.clone().detach())
        if self.aux_loss:
            aux_loss = out["aux"]
            aux_loss = torch.mean(aux_loss)
            total_loss = aux_loss + total_loss
            self.val_loss_values["aux"].append(aux_loss.clone().detach())
        
        pred = torch.softmax(pred_logits, dim=1)           # [b, num_cls, h, w]
        pred = pred > self.conf_thresh
        
        self.val_metrics.update(pred, masks)
        return total_loss
    
    def validation_epoch_end(self, losses):
        # if isinstance(losses[0], dict):
        #     losses = [o["loss"] for o in losses]
        # focal_losses, aux_losses = zip(*losses)
        # focal_losses = torch.Tensor(focal_losses).to(self.device)

        for key, values in self.val_loss_values.items():
            tensor = torch.Tensor(values).to(self.device)
            self.sync_and_log_losses_on_epoch(tensor, f"avg_loss/val/{key}")

        self.val_loss_values = {k: [] for k in self.val_loss_values.keys()}

        miou = self.val_metrics.jaccard(average=False)
        dice = self.val_metrics.dice(average=False)
        acc = self.val_metrics.accuracy(average=False)

        self.log("mean_iou/val/background", miou[0], rank_zero_only=True)
        self.log("mean_iou/val/reflection", miou[1], rank_zero_only=True)
        self.log("mean_iou/val/avg", torch.mean(miou), rank_zero_only=True)
        
        self.log("dice/val/background", dice[0], rank_zero_only=True)
        self.log("dice/val/reflection", dice[1], rank_zero_only=True)
        self.log("dice/val/avg", torch.mean(dice), rank_zero_only=True)

        self.log("accuracy/val/background", acc[0], rank_zero_only=True)
        self.log("accuracy/val/reflection", acc[1], rank_zero_only=True)
        self.log("accuracy/val/avg", torch.mean(acc), rank_zero_only=True)

        self.val_metrics.reset()
    
    def sync_and_log_losses_on_epoch(self, loss_tensor: torch.Tensor, name: str):
        tensor_list = [torch.zeros_like(loss_tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list, loss_tensor)
        tensor_list = torch.stack(tensor_list, dim=0)
        avg_loss = torch.mean(tensor_list)
        self.log(name, avg_loss.item(), rank_zero_only=True, on_epoch=True)
    
    def forward(self, x):
        return self.model(x)


def train(
    train_dir: str,
    val_dir: str,
    test_dir: str,
    mask_subdir: str,
    out_dir: str,
    backbone: str,
    num_epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    weight_decay: float,
    gamma: float,
    betas: List[float],
    aux_loss: bool,
    auto_class_weights: bool,
    num_classes:int = 2,
    lr_min: float = 0.00001,
    no_empty_masks: bool = False,
    img_size: List[int] = [480, 960],
    run_name: str = None
):

    assert backbone in BACKBONE_LOOKUP.keys(), f"backbone must be one of {BACKBONE_LOOKUP.keys()}"

    # seeds for reproducibility
    torch.random.manual_seed(123)
    np.random.seed(123)
    random.seed(123)

    current = datetime.now()
    timestamp = current.strftime("%Y-%m-%d_%H-%M-%S")

    run_name += "_" + timestamp

    out_dir = os.path.join(out_dir, run_name)
    # setup logger
    logger_dir = os.path.join(out_dir, "output")
    logger = pl_loggers.WandbLogger(project="ai4od-semseg-poc-paper", name=run_name, log_model=False, save_dir=logger_dir)

    logger.log_hyperparams(
        {
            "train_dir": train_dir,
            "val_dir": val_dir,
            "test_dir": test_dir,
            "mask_subdir": mask_subdir,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "img_size": img_size,
            "aux_loss": aux_loss,
            "auto_class_weights": auto_class_weights,
            "lr_min": lr_min
        }
    )

    # setup checkpoint logging
    ckp_dir = os.path.join(out_dir, "model")
    ckp_callback_iou = ModelCheckpoint(monitor="mean_iou/val/reflection", mode="max", dirpath=ckp_dir, filename="best_iou.ckpt")
    ckp_callback_dice = ModelCheckpoint(monitor="dice/val/reflection", mode="max", dirpath=ckp_dir, filename="best_dice.ckpt")
    ckp_callback_acc = ModelCheckpoint(monitor="accuracy/val/reflection", mode="max", dirpath=ckp_dir, filename="best_acc.ckpt")
    ckp_callback_loss = ModelCheckpoint(monitor="avg_loss/val/total", mode="min", dirpath=ckp_dir, filename="best_loss.ckpt")
    lr_monitor = LearningRateMonitor("epoch")
    
    callbacks = [ckp_callback_iou, ckp_callback_dice, ckp_callback_loss, ckp_callback_acc, lr_monitor]

    # setup lightning modules
    filters = []
    if no_empty_masks:
        filters.append(EmptyFilter())

    data_module = SemanticDataModule(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers,
        img_subdir="images",
        mask_subdir=mask_subdir,
        filters=filters
    )

    # calculate the class weights based on the train dataset
    class_weights = None
    if auto_class_weights:
        class_weights = calculate_class_weights(dataloader=data_module.train_dataloader(), num_classes=num_classes, sync=False)

        # class_weights = torch.Tensor([0.01, 1.]).to("cuda")
        class_weights *= 1 / class_weights.min()
        
    print(f"Class Weights:", class_weights)

    model_fn, backbone_weights = BACKBONE_LOOKUP[backbone]
    model = model_fn(num_classes=2, aux_loss=aux_loss, backbone_weights=backbone_weights)

    network_module = LightningNetwork(
        model=model,
        num_classes=num_classes,
        lr=lr,
        wd=weight_decay,
        betas=betas,
        gamma=gamma,
        class_weights=class_weights,
        eta_min=lr_min,
        tmax=num_epochs
    )

    # start training
    device_ids = list(range(torch.cuda.device_count()))    # all visible devices
    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=False),
        accelerator="gpu",
        devices=device_ids,
        max_epochs=num_epochs,
        log_every_n_steps=20,
        sync_batchnorm=True,
        logger=logger,
        callbacks=callbacks
    )

    # this is a hack because we need to wait until here so that the process groups are initialized by lightning
    trainer.fit(network_module, data_module)
    
    # cleanup wandb
    wandb.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--learning-rate-min", type=float, default=0.00001)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=4.)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--aux-loss", type=bool, default=False)
    parser.add_argument("--class-weights", action='store_true')

    parser.add_argument("--backbone", type=str, default="resnet50", choices=list(BACKBONE_LOOKUP.keys()))
    parser.add_argument("--no-empty-masks", action="store_true", help="Flag whether to exclude images without annotated reflections from training.")

    parser.add_argument("--train-dir", type=str, default=os.environ.get("TRAIN_DIR"))
    parser.add_argument("--val-dir", type=str, default=os.environ.get("VAL_DIR"))
    # parser.add_argument("--test-dir", type=str, default=os.environ.get("TEST_DIR"))
    parser.add_argument("--mask-subdir", type=str, default="masks/intersection")

    parser.add_argument("--out-dir", type=str, default="default_out")

    parser.add_argument("--run-name", type=str, default="default_run")

    args = parser.parse_args()

    # args.train_dir = "/home/ubuntu/ai4od-semseg-paper/data/5-fold/test"
    # args.val_dir = "/home/ubuntu/ai4od-semseg-paper/data/5-fold/test"
    # args.test_dir = "/home/ubuntu/ai4od-semseg-paper/data/5-fold/test"
    # args.out_dir = "/home/ubuntu/ai4od-semseg-paper/out"

    train(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=None,
        mask_subdir=args.mask_subdir,
        out_dir=args.out_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        num_workers=os.cpu_count(),
        weight_decay=args.weight_decay,
        gamma=args.gamma,
        betas=[args.beta1, args.beta2],     
        aux_loss=args.aux_loss,
        run_name=args.run_name,
        auto_class_weights=args.class_weights,
        lr_min=args.learning_rate_min,
        backbone=args.backbone,
        no_empty_masks=args.no_empty_masks,
        num_classes=2,                      # fixed
        img_size=[240, 960]                 # fixed
    )
