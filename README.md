# [ECCV 2024] T-MAE: Temporal Masked Autoencoders for Point Cloud Representation Learning
---
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2312.10217)
[![Website](doc/badges/badge-website.svg)](https://codename1995.github.io/t-mae.github.io/)



## 🏠 About
This repository contains the official implementation of the paper [T-MAE: Temporal Masked Autoencoders for Point Cloud Representation Learning](https://arxiv.org/abs/2312.10217) by Weijie Wei, Fatemeh Karimi Najadasl, Theo Gevers and Martin R. Oswald.

## 🔥 News
- [2024/09/19] The code will be released soon.
- [2024/09/22] Release the code of evaluation on ONCE dataset.

## Table of Contents

## TODO
- [x] Release ONCE evaluation code.
- [ ] Release ONCE training code.
- [ ] Release Waymo training code and inference code.

## Installation
We test this environment with NVIDIA A100 GPUs and Linux RHEL 8.

```
conda create -n t-mae python=3.8
conda activate t-mae
conda install -y pytorch==1.11.0 torchvision==0.12.0 torchaudio=0.11.0 cudatoolkit=11.3 -c pytorch
conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.1"
pip install numpy==1.19.5 protobuf==3.19.4 scikit-image==0.19.2 spconv-cu113 numba scipy pyyaml easydict fire tqdm shapely matplotlib opencv-python addict pyquaternion awscli open4d pandas future pybind11 tensorboardX tensorboard Cython 
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html

pip install pycocotools
pip install SharedArray
pip install tensorflow-gpu==2.5.0
pip install protobuf==3.20

git clone https://github.com/codename1995/T-MAE
cd T-MAE && python setup.py develop --user
cd pcdet/ops/dcn && python setup.py develop --user
```

## Data Preparation
Please follow the [instruction](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md) of OpenPCDet to prepare the dataset. For the Waymo dataset, we use the [evaluation toolkits](https://drive.google.com/drive/folders/1aa1kI9hhzBoZkIBcr8RBO3Zhg_RkOAag?usp=sharing) to evaluate detection results, where the `compute_detection_metrics_main` file comes from Waymo-open-dataset API ([Mar 2023](https://github.com/waymo-research/waymo-open-dataset/blob/9d0be7c4aeed2ef5e11f2e739537bf2d27cec65b/docs/quick_start.md)) and its source code](https://github.com/waymo-research/waymo-open-dataset/blob/9d0be7c4aeed2ef5e11f2e739537bf2d27cec65b/src/waymo_open_dataset/metrics/tools/compute_detection_metrics_main.cc) is C++ based.

```
data
│── waymo
│   │── ImageSets/
│   │── raw_data
│   │   │── segment-xxxxxxxx.tfrecord
│   │   │── ...
│   │── waymo_processed_data
│   │   │── segment-xxxxxxxx/
│   │   │── ...
│   │── waymo_processed_data_gt_database_train_sampled_1/
│   │── waymo_processed_data_waymo_dbinfos_train_sampled_1.pkl
│   │── waymo_processed_data_infos_test.pkl
│   │── waymo_processed_data_infos_train.pkl
│   │── waymo_processed_data_infos_val.pkl
│   │── compute_detection_metrics_main
│   │── gt.bin
│── once
│   │── ImageSets/
│   │── data
│   │   │── 000000/
│   │   │── ...
│   │── gt_database/
│   │── once_dbinfos_train.pkl
│   │── once_infos_raw_large.pkl
│   │── once_infos_raw_medium.pkl
│   │── once_infos_raw_small.pkl
│   │── once_infos_train.pkl
│   │── once_infos_val.pkl
│── ckpts
│   │── once_tmae_weights.pth
│   │── ...
```


## Training & Testing

```
# t-mae pretrain
bash scripts/once_ssl_train.sh

# finetune
bash scripts/once_train.sh

# test
bash scripts/once_test.sh
```

## Results
### Waymo
Reproduced results to be updated soon.
We could not provide the above pretrained models due to [Waymo Dataset License Agreement](https://waymo.com/open/terms/).


### ONCE
| | mAP | Vehicle | Pedestrian | Cyclist | Pretrained Weights |
| --- | :---: | :---: | :---: | :---: | :---: |
| T-MAE | 67.41 | 77.53 | 54.81 | 69.90 | [ckpt](https://drive.google.com/file/d/1_8YrjzobyxrK86TyQphGZwhEjBMOfnOa/view?usp=drive_link) |


## Citation
If you find this repository useful, please consider citing our paper.

```
@inproceedings{wei2024tmae,
  title={T-MAE: Temporal Masked Autoencoders for Point Cloud Representation Learning},
  author={Weijie Wei, Fatemeh Karimi Najadasl, Theo Gevers and Martin R. Oswald},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```


## Acknowledgements
This project is mainly based on the following repositories:
- [GD-MAE](https://github.com/nightmare-n/GD-MAE)
- [SST](https://github.com/tusen-ai/SST)
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)

We would like to thank the authors for their great work.

