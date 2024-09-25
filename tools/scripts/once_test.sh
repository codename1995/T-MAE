#!/usr/bin/env bash

NGPUS=4

CFG_NAME=once_models/t_mae
TAG_NAME=tmae_pretrained

python -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_port 12345 test.py \
  --launcher pytorch --cfg_file cfgs/$CFG_NAME.yaml --workers 8 --extra_tag $TAG_NAME \
  --ckpt ../ckpt/once_tmae_weights.pth --fixed_gap_eval 1