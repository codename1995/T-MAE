#!/usr/bin/env bash

PYTHON="/home/wwei2/miniconda3/envs/gd-mae/bin/python"
NGPUS=4

# T-MAE Pretraining
SSL_CFG_NAME=once_models/t_mae_ssl
SSL_TAG_NAME=default

python -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_port 12345 train.py --launcher \
  pytorch --cfg_file cfgs/$SSL_CFG_NAME.yaml --workers 16 --extra_tag $SSL_TAG_NAME --max_ckpt_save_num 1 \
  --num_epochs_to_eval 0 --amp

# Fine-tuning
CFG_NAME=once_models/t_mae
TAG_NAME=tmae_pretrained_$SSL_TAG_NAME

python -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_port 12345 train.py \
  --launcher pytorch --cfg_file cfgs/$CFG_NAME.yaml --workers 8 --extra_tag $TAG_NAME \
  --max_ckpt_save_num 1 --num_epochs_to_eval 1 --amp --fixed_gap_eval 1 \
  --pretrained_model ../output/$SSL_CFG_NAME/$SSL_TAG_NAME/ckpt/checkpoint_epoch_12.pth