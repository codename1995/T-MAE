import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt
from easydict import EasyDict

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file#, load_base_config
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model
import copy
import wandb
import pdb
from torch.cuda.amp import GradScaler, autocast

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
# os.environ['CUDA_VISIBLE_DEVICES']='8,9'
# os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7'

# class PathDecider()     def __init__(self, running_device, cfg, **kwargs):
#         if running_device == "das":
#             self.output_dir = Path('/var/scratch/wwei2/GD-MAE/')
#             self.data_base_dir = '/var/scratch/wwei2/Datasets/'
#         elif running_device == "laptop":
#             self.output_dir = cfg.ROOT_DIR
#             self.data_base_dir = '/home/lwei/Datasets/'
#         elif running_device == "desktop":
#             self.output_dir = cfg.ROOT_DIR
#             self.data_base_dir = '../data'
#         elif running_device == "deepdrivers":
#             self.output_dir = cfg.ROOT_DIR
#             self.data_base_dir = '../data'
#         else:
#             raise NotImplementedError
#
#     def get_output_dir(self):
#         return self.output_dir
#
#     def update_cfg(self, cfg):
#         cfg.DATA_CONFIG.DATA_PATH = self.data_base_dir + cfg.DATA_CONFIG.DATA_PATH
#         return cfg
#

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False,
                        help='if true, set len(dataset) to #samples * #epochs')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=1, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=10, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--fuse_conv_bn', action='store_true', default=False, help='')

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--wandb', action='store_true', default=False, help='whether to use wandb')
    parser.add_argument('--wandb_proj_name', type=str, default='t-mae-0.05',
                        help='whether to use wandb')
    parser.add_argument('--record_batch', action='store_true', default=False, help='whether to use record frame_id for current batch')
    parser.add_argument('--fixed_gap', type=int, default=-1, help='fixed gap between frames')
    parser.add_argument('--amp', action='store_true', default=False, help='whether to use AMP (automatic mixed precision)')
    parser.add_argument('--partially_ft', action='store_true', default=False, help='whether to use partially finetune')


    # parser.add_argument('--running_device', type=str, default="das",
    #                     choices=["das", "laptop", "desktop"],
    #                     help='which cluster to use. Modify the path to data and to save ckpt and log. ')


    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    # if hasattr(cfg, '_BASE_CONFIG'):
    #     load_base_config(cfg._BASE_CONFIG, cfg)

    args.sync_bn = args.sync_bn or cfg.OPTIMIZATION.get('SYNC_BN', False)
    
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666)

    if args.local_rank ==0 and args.wandb:
        wandb.login()
        temp_cfgs = copy.deepcopy(cfg)
        temp_cfgs.update(args.__dict__)
        wandb.init(project=args.wandb_proj_name, name=args.extra_tag, config=temp_cfgs)
        # del temp_cfgs

    # pather = PathDecider(args.running_device, cfg)
    # cfg = pather.update_cfg(cfg)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    # output_dir = pather.output_dir / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    # ckpt_dir = Path(output_dir) / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # -----------------------create dataloader & network & optimizer---------------------------
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set, logger=logger)
    # total_params = sum(
    #     param.numel() for param in model.parameters()
    # )

    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)


    scaler = GradScaler() if args.amp else None

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)

    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger, scaler=scaler)
        last_epoch = start_epoch + 1
    else:
        # Automatically resume from the lastest checkpoint
        ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            it, start_epoch = model.load_params_with_optimizer(
                ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger, scaler=scaler
            )
            last_epoch = start_epoch + 1

    if args.partially_ft:
        for params in model.vfe.parameters():
            params.requires_grad = False
        for params in model.backbone_3d.sst_blocks.parameters():
            params.requires_grad = False
        for params in model.backbone_3d.ca_blocks.parameters():
            params.requires_grad = False

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()], find_unused_parameters=False)
    logger.info(model)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader)//args.gradient_accumulation_steps,
        total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # -----------------------start training---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    # train_model(
    #     model,
    #     optimizer,
    #     train_loader,
    #     model_func=model_fn_decorator(),
    #     lr_scheduler=lr_scheduler,
    #     optim_cfg=cfg.OPTIMIZATION,
    #     start_epoch=start_epoch,
    #     total_epochs=args.epochs,
    #     start_iter=it,
    #     rank=cfg.LOCAL_RANK,
    #     tb_log=tb_log,
    #     ckpt_save_dir=ckpt_dir,
    #     train_sampler=train_sampler,
    #     lr_warmup_scheduler=lr_warmup_scheduler,
    #     ckpt_save_interval=args.ckpt_save_interval,
    #     max_ckpt_save_num=args.max_ckpt_save_num,
    #     merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
    #     gradient_accumulation_steps=args.gradient_accumulation_steps,
    #     use_wandb=args.wandb,
    #     record_batch=args.record_batch,
    #     scaler=scaler,
    # )

    if hasattr(train_set, 'use_shared_memory') and train_set.use_shared_memory:
        train_set.clean_shared_memory()

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    logger.info('**********************Start evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    if args.fixed_gap >= 2:
        cfg.DATA_CONFIG.FIXED_GAP = args.fixed_gap
    elif args.fixed_gap == 1:
        cfg.DATA_CONFIG.SCAN_WINDOW_TST = 2
    elif args.fixed_gap == 0:
        cfg.DATA_CONFIG.SCAN_WINDOW_TST = 2
        cfg.DATA_CONFIG.SAME_INPUT = True
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers, logger=logger, training=False
    )
    if args.fixed_gap >= 0:
        eval_output_dir = output_dir / 'eval' / ('eval_with_train_%d' % args.fixed_gap)
    else:
        eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    args.start_epoch = max(args.epochs - args.num_epochs_to_eval + 1, 0)  # Only evaluate the last 10 epochs

    repeat_eval_ckpt(
        model.module if dist_train else model,
        test_loader, args, eval_output_dir, logger, ckpt_dir,
        dist_test=dist_train,
        use_wandb=args.wandb,
    )
    logger.info('**********************End evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    if args.local_rank == 0 and args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()