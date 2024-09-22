import argparse
import datetime
import os
from pathlib import Path

import torch.cuda
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from eval_utils import eval_utils
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
import torch.nn.functional as F

from tools.visual_utils.open3d_vis_utils import draw_scenes, draw_scenes_with_2pcs

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

def remove_points_outside_range(points, coords_range):
    """Remove points out of predefined range.
    Args:
        points (np.ndarray): Input points.
        coords_range (list[float]): Range of point cloud.
    Returns:
        np.ndarray: Points in the range.
    """
    assert points.shape[1] == 3
    mask = (points[:, 0] >= coords_range[0]) & (points[:, 0] <= coords_range[3]) & \
           (points[:, 1] >= coords_range[1]) & (points[:, 1] <= coords_range[4]) & \
           (points[:, 2] >= coords_range[2]) & (points[:, 2] <= coords_range[5])
    points = points[mask]
    return points

def get_pixel_index(points, coords_range, voxel_size):
    x_idx = np.array((points[:, 0] - coords_range[0]) // voxel_size[0], dtype=np.int32)
    y_idx = np.array((points[:, 1] - coords_range[1]) // voxel_size[1], dtype=np.int32)
    return [x_idx, y_idx]

def convert_attn_to_heatmap(attn_map, cmap_name='viridis'):
    attn_map = attn_map / attn_map.max()

    cmap = get_cmap(cmap_name)
    heatmap = cmap(attn_map)
    return heatmap[:, :, :3]

def get_color(idx_2d, heatmap):
    return heatmap[idx_2d[1], idx_2d[0]]

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str,
                        default="./cfgs/waymo_temporal_models/eccv_resub_final_vis.yaml",
                        help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    # parser.add_argument('--pretrained_model', type=str, default="../ckpt/no_pretrained_ft100_for_ep30.pth", help='pretrained_model')
    # parser.add_argument('--pretrained_model', type=str, default="../ckpt/tmae_pretrain_mvjar_ep12.pth", help='pretrained_model')
    parser.add_argument('--pretrained_model', type=str, default="../ckpt/tmae_pretrained_ft100_ep30.pth", help='pretrained_model')
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

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=10, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--fuse_conv_bn', action='store_true', default=False, help='')

    parser.add_argument('--fixed_gap', type=int, default=3, help='fixed gap between frames')
    parser.add_argument('--amp', action='store_true', default=False, help='whether to use AMP (automatic mixed precision)')
    parser.add_argument('--partially_ft', action='store_true', default=False, help='whether to use partially finetune')

    # parser.add_argument('--cmap', type=str, default='hot', help='colormap for attention map')
    parser.add_argument('--cmap', type=str, default='hot', help='colormap for attention map')

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

    # pather = PathDecider(args.running_device, cfg)
    # cfg = pather.update_cfg(cfg)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_output_dir = output_dir / 'eval'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # -----------------------create dataloader & network & optimizer---------------------------
    if args.fixed_gap >= 2:
        cfg.DATA_CONFIG.FIXED_GAP = args.fixed_gap
    elif args.fixed_gap == 1:
        cfg.DATA_CONFIG.SCAN_WINDOW_TST = 2

    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=False,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set, logger=logger)
    model.cuda()

    # load checkpoint if it is possible
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)

    logger.info(model)

    logger.info('**********************Start Inference %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    model.eval()
    voxel_size=np.array([0.32, 0.32, 6.0])
    coords_range=np.array([-74.88, -74.88, -2, 74.88, 74.88, 4.0])
    frame_gap = 50 if "waymo" in args.cfg_file else 20  # 50 for waymo, 20 for ONCE
    save_prefix = "" if "waymo" in args.cfg_file else "once"
    for i, batch_dict in enumerate(train_loader):
        if i % frame_gap != 0:
            continue
        # if i == 200 or i == 500:
        #     pass
        # else:
        #     continue
        points = batch_dict['points'][:, 1:4]
        gt_boxes = batch_dict['gt_boxes'][0]

        torch.cuda.synchronize()
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        torch.cuda.synchronize()

        pred_box = pred_dicts[0]['pred_boxes'].cpu().numpy()

        draw_scenes(points, gt_boxes=gt_boxes, ref_boxes=pred_box, save_image=True,
                    image_path='%s_%s.png'%(save_prefix, i))

        print("%d / %d"%(i, len(train_loader)))
        pass



    logger.info('**********************End evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()
