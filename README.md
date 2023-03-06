# Detecting Oncoming Vehicles at Night in Urban Scenarios - An Annotation Proof-Of-Concept

This repository will contains the code for the paper "Detecting Oncoming Vehicles at Night in Urban Scenarios - An Annotation Proof-Of-Concept".

![](method_figure.png)

## Dataset

The dataset is available [here on kaggle](https://www.kaggle.com/datasets/lukasewecker/urban-provident-vehicle-detection-at-night). Please download the dataset before you continue with the next steps. 

The dataset only contains the images and original bounding box annotations. To generate the binary masks, follow the next steps.

The data is stored scene-wise. This means that the data for each recorded scene is stored in a separate directory. Each scene directory then contains an `image` and several `annotations` folders. For example, this is the structure of scene 3:

```
00003
    --- images
        --- image1.jpg
        --- image2.jpg
        --- ....
        --- imageN.jpg
    --- annotations1
        --- image1.json
        --- image2.json
        --- ....
        --- imageN.json
    --- annotations5
        --- image1.json
        --- image2.json
        --- ....
        --- imageN.json
    --- annotations8
        --- image1.json
        --- image2.json
        --- ....
        --- imageN.json
```

The scene was annotated from annotators 1, 5, and 8, thus containing the three distinct annotation directories.


## Installation

This repo requires the pytorch & torchvision nightly builds. To install, do the following:

First, install PyTorch nightly:
```
python3 -m venv venv
source venv bin activate

pip install numpy

# adjust cuda version if necessary
pip install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cu117

```

Install `torchvision`:
```
git clone https://github.com/pytorch/vision.git
cd vision
git checkout d2d448c71b4cb054d160000a0f63eecad7867bdb
python setup.py develop
pip install flake8 typing mypy pytest pytest-mock scipy
```

Install this repo as a package:
```
cd ..
pip install -e .
```

If you want to train the models, login to Weights & Biases:
```
wandb login
```

## Create the segmentation masks
```
python3 scripts/create_mask_dataset.py --param-path dataset/best_config.json --data-dir <path/to/dataset/> --mask-subdir majority-best-params --aggregation majority
```

## Create the 5-fold cross validation dataset
```
python3 scripts/create_k_fold_dataset.py --data-dir <path/to/dataset/> --out-dir <path/to/5-fold-dataset/> --k 5 
```

## Train the model

```
python training/train_kfold.py --data-dir <path/to/5-fold-dataset/> --num-epochs 200 --learning-rate 0.001 --gamma 4 --out-dir out/ --run-name 5fold-resnet50 --mask-subdir masks/majority-best-params --backbone resnet50 --no-empty-masks
```

## Reproduce annotation results

[This notebook](notebooks/annotations.ipynb) walks you through reproducing all the numbers from the analysis of the annotation PoC in the paper.

## Reproduce the segmentation method parameter search

The scripts to perform a grid search for all the four segmentation methods mentiond in the paper are located in [scripts/grid_search/](scripts/grid_search/). For example:

```
cd scripts/grid_search/
python3 grid_search_adapt_gaussian.py --data-dir <path/to/dataset>
```
