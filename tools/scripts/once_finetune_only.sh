#!/usr/bin/env bash
NGPUS=4

# Fine-tuning
CFG_NAME=once_models/t_mae
TAG_NAME=tmae_pretrained_$SSL_TAG_NAME


# Load provided pretrained model and finetune on ONCE
python -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_port 12345 train.py \
  --launcher pytorch --cfg_file cfgs/$CFG_NAME.yaml --workers 8 --extra_tag $TAG_NAME \
  --max_ckpt_save_num 1 --num_epochs_to_eval 1 --amp --fixed_gap_eval 1 \
  --pretrained_model ../ckpt/once_tmae_pretrained.pth 