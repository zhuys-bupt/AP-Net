# AP-Net

This is the implementation of the paper **AP-Net: Attention-Fused Volume and Progressive Aggregation for
Accurate Stereo Matching**.

# How to use

## Environment
* python 3.6
* Pytorch >= 1.5.1

## Data Preparation
Download [Scene Flow Datasets](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo), [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

## Training
**Scene Flow Datasets**

run the script `./scripts/sceneflow.sh` to train on Scene Flow datsets. Please update `DATAPATH` in the bash file as your training data path.

**KITTI 2012 / 2015**

run the script `./scripts/kitti12.sh` and `./scripts/kitti15.sh` to finetune on the KITTI 12/15 dataset. Please update `DATAPATH` and `--loadckpt` as your training data path and pretrained SceneFlow checkpoint file.

## Evaluation
run the script `./scripts/kitti12_save.sh` and `./scripts/kitti15_save.sh` to save png predictions on the test set of the KITTI datasets to the folder you specify.

# Citation
If you find this code useful in your research, please cite:
@article{ZHU2024128685,
title = {AP-Net: Attention-fused volume and progressive aggregation for accurate stereo matching},
journal = {Neurocomputing},
pages = {128685},
year = {2024},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2024.128685},
url = {https://www.sciencedirect.com/science/article/pii/S0925231224014565},
author = {Yansong zhu and Songwei Pei and BingFeng Liu and Jun Gao},
keywords = {Attention-fused cost volume, Progressive cost aggregation, Refinement module},
}


# Acknowledgements

Special thanks to authors of [GwcNet](https://github.com/xy-guo/GwcNet) and [STTR](https://github.com/mli0603/stereo-transformer) for open-sourcing the code.
