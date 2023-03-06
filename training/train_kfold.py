import os
from typing import List

from training.train import train
from training.train import BACKBONE_LOOKUP


def train_kfold(
    data_dir: str,
    mask_subdir: str,
    out_dir: str,
    num_epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    weight_decay: float,
    gamma: int,
    betas: List[float],
    aux_loss: bool,
    backbone: str,
    no_empty_masks: bool,
    k: List[int] = None,
    num_classes:int = 2,
    img_size: List[int] = [480, 960],
    run_name: str = None,
    auto_class_weights: bool = False,
    lr_min: float = 0.00001
    ):

    k = k if k else list(range(5))
    for i in k:
        train_dir = os.path.join(data_dir, f"{i}", "train")
        val_dir = os.path.join(data_dir, f"{i}", "val")
        test_dir = os.path.join(data_dir, "test")

        print("Training on", train_dir)
        print("Validating on", val_dir)

        k_out_dir = os.path.join(out_dir, str(i))

        k_run_name = f"{run_name}_{i}" if run_name else None
        
        train(
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            mask_subdir=mask_subdir,
            backbone=backbone,
            out_dir=k_out_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            num_workers=num_workers,
            weight_decay=weight_decay,
            gamma=gamma,
            betas=betas,
            aux_loss=aux_loss,
            run_name=k_run_name,
            auto_class_weights=auto_class_weights,
            lr_min=lr_min,
            no_empty_masks=no_empty_masks,
            num_classes=num_classes,
            img_size=img_size,
        )


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
    parser.add_argument("--no-empty-masks", action="store_true", help="Flag whether to exclude images without annotated reflections from training.")

    parser.add_argument("--backbone", type=str, default="resnet50", choices=list(BACKBONE_LOOKUP.keys()))

    parser.add_argument("-k", nargs="+", type=int, default=None)

    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--mask-subdir", type=str, default="masks/intersection")

    parser.add_argument("--out-dir", type=str)

    parser.add_argument("--run-name", type=str, default="default_run")

    args = parser.parse_args()

    train_kfold(
        data_dir=args.data_dir,
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
        k=args.k,
        no_empty_masks=args.no_empty_masks,
        num_classes=2,                      # fixed
        img_size=[480, 960]                 # fixed
    )
