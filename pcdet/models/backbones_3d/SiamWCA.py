import numpy as np
import torch
import torch.nn as nn
from ..model_utils.sst_basic_block import BasicShiftBlockV2
from ..model_utils.wca_block import BasicShiftBlock_WCA
from ..model_utils import sst_utils
from ...ops.sst_ops import sst_ops_utils
from functools import partial
from ...utils.spconv_utils import (
    replace_feature,
    spconv,
    post_act_block,
    SparseBasicBlock,
    post_act_block_GN,
)

from .spt_backbone import SSTInputLayer, SSTBlockV1
import pdb


class SSTInputLayer_Temporal(SSTInputLayer):
    """
    This is one of the core class of SST, converting the output of voxel_encoder to sst input.
    There are 3 things to be done in this class:
    1. Reginal Grouping : assign window indices to each voxel.
    2. Voxel drop and region batching: see our paper for detail
    3. Pre-computing the transfomation information for converting flat features ([N x C]) to region features ([R, T, C]).
        R is the number of regions containing at most T tokens (voxels). See function flat2window and window2flat for details.

    Main args:
        drop_info (dict): drop configuration for region batching.
        window_shape (tuple[int]): (num_x, num_y). Each window is divided to num_x * num_y pillars (including empty pillars).
        shift_list (list[tuple]): [(shift_x, shift_y), ]. shift_x = 5 means all windonws will be shifted for 5 voxels along positive direction of x-aixs.
        debug: apply strong assertion for developing.
    """

    def __init__(self, model_cfg, **kwargs):
        super().__init__(model_cfg, **kwargs)

    def _padding_bincount(self, bincount1, bincount2):
        l1 = bincount1.shape[0]
        l2 = bincount2.shape[0]

        if l1 > l2:
            bincount2 = torch.cat(
                (
                    bincount2,
                    torch.zeros(
                        l1 - l2, dtype=bincount2.dtype, device=bincount2.device
                    ),
                )
            )
        elif l1 < l2:
            bincount1 = torch.cat(
                (
                    bincount1,
                    torch.zeros(
                        l2 - l1, dtype=bincount1.dtype, device=bincount1.device
                    ),
                )
            )

        return bincount1, bincount2

    def drop_single_shift_ref_to_prv(self, batch_win_inds, batch_win_inds_prv):
        # Inherited from drop_single_shift, the differences are
        # 1. drop_single_shift() returns drop_lvl_per_voxel but barely drop voxels, while this
        # func also returns drop_lvl_per_voxel but drops a lot of voxels.
        # 2. voxels are dropped if no voxels in windows of either current or previous frame
        # 3. drop_lvl_per_voxel is based on the larger number of voxels per window
        drop_info = self.drop_info

        drop_lvl_per_voxel = -torch.ones_like(
            batch_win_inds
        )  # drop_lvl_per_voxel: Tensor: (11397,)
        drop_lvl_per_voxel_prv = -torch.ones_like(batch_win_inds_prv)
        bincount = torch.bincount(batch_win_inds)  # key=window_id, value=num_voxels
        bincount_prv = torch.bincount(batch_win_inds_prv)

        bincount, bincount_prv = self._padding_bincount(bincount, bincount_prv)

        # bincount_no_voxel = bincount == 0
        # a = torch.sum(bincount_no_voxel)  # a: tensor(24454, device='cuda:0')
        # bincount_no_voxel_prv = bincount_prv == 0
        # f = torch.sum(bincount_no_voxel_prv)  # f:
        bincount_no_voxel = torch.logical_or(bincount == 0, bincount_prv == 0)
        no_voxel_mask = bincount_no_voxel[
            batch_win_inds
        ]  # no voxels in this window, delete this
        # b = torch.sum(no_voxel_mask)  # b: tensor(0, device='cuda:0')
        no_voxel_mask_prv = bincount_no_voxel[batch_win_inds_prv]
        # c = torch.sum(no_voxel_mask_prv)  # c: tensor(19615, device='cuda:0')

        bincount_max = torch.cat(
            (bincount.unsqueeze(1), bincount_prv.unsqueeze(1)), dim=1
        )
        bincount_max = torch.max(bincount_max, dim=1).values

        num_per_voxel_before_drop = bincount_max[batch_win_inds]
        num_per_voxel_before_drop_prv = bincount_max[batch_win_inds_prv]
        target_num_per_voxel = torch.zeros_like(batch_win_inds)
        target_num_per_voxel_prv = torch.zeros_like(batch_win_inds_prv)

        for dl in drop_info:
            max_tokens = drop_info[dl]["max_tokens"]
            lower, upper = drop_info[dl]["drop_range"]

            range_mask = (num_per_voxel_before_drop >= lower) & (
                num_per_voxel_before_drop < upper
            )
            target_num_per_voxel[range_mask] = max_tokens
            drop_lvl_per_voxel[range_mask] = dl

            range_mask_prv = (num_per_voxel_before_drop_prv >= lower) & (
                num_per_voxel_before_drop_prv < upper
            )
            target_num_per_voxel_prv[range_mask_prv] = max_tokens
            drop_lvl_per_voxel_prv[range_mask_prv] = dl

        assert (target_num_per_voxel > 0).all()
        assert (target_num_per_voxel_prv > 0).all()
        assert (drop_lvl_per_voxel >= 0).all()
        assert (drop_lvl_per_voxel_prv >= 0).all()

        inner_win_inds = sst_ops_utils.get_inner_win_inds(batch_win_inds)
        inner_win_inds_prv = sst_ops_utils.get_inner_win_inds(batch_win_inds_prv)
        keep_mask = torch.logical_and(
            inner_win_inds < target_num_per_voxel, torch.logical_not(no_voxel_mask)
        )
        # d = torch.sum(keep_mask)  # d: tensor(11397, device='cuda:0')
        keep_mask_prv = torch.logical_and(
            inner_win_inds_prv < target_num_per_voxel_prv,
            torch.logical_not(no_voxel_mask_prv),
        )
        # e = torch.sum(keep_mask_prv)  # e: tensor(58819, device='cuda:0')
        # Since sst_ops_utils.get_inner_win_inds(batch_win_inds) is not understandable (written in .cu),
        # I can only guess the meaning of inner_win_inds. I think it is the index of a voxel in its
        # belonging window.
        # In conclusion, last line means deleting the last voxel in each window.
        return keep_mask, drop_lvl_per_voxel, keep_mask_prv, drop_lvl_per_voxel_prv

    def drop_voxel(self, voxel_info, voxel_info_prv, num_shifts=2):
        """
        To make it clear and easy to follow, we do not use loop to process two shifts.
        """

        for i in range(num_shifts):
            batch_win_inds = voxel_info[f"batch_win_inds_shift{i}"]
            num_all_voxel = batch_win_inds.shape[0]

            batch_win_inds_prv = voxel_info_prv[f"batch_win_inds_shift{i}"]
            num_all_voxel_prv = batch_win_inds_prv.shape[0]

            voxel_keep_inds = torch.arange(
                num_all_voxel, device=batch_win_inds.device, dtype=torch.long
            )
            voxel_keep_inds_prv = torch.arange(
                num_all_voxel_prv, device=batch_win_inds_prv.device, dtype=torch.long
            )

            keep_mask, drop_lvl, keep_mask_prv, drop_lvl_prv = (
                self.drop_single_shift_ref_to_prv(batch_win_inds, batch_win_inds_prv)
            )

            # a = torch.unique(batch_win_inds[keep_mask], sorted=True)
            # b = torch.unique(batch_win_inds_prv[keep_mask_prv], sorted=True)
            # assert len(a) == len(b)

            # voxel_num_before_drop = len(voxel_info['voxel_coords'])
            voxel_info[f"voxel_keep_inds_shift{i}"] = voxel_keep_inds[keep_mask]
            voxel_info[f"voxel_drop_level_shift{i}"] = drop_lvl[keep_mask]
            voxel_info[f"batch_win_inds_shift{i}"] = batch_win_inds[keep_mask]
            voxel_info[f"voxel_features_shift{i}"] = voxel_info["voxel_features"][
                keep_mask
            ]
            voxel_info[f"voxel_coords_shift{i}"] = voxel_info["voxel_coords"][keep_mask]
            voxel_info[f"coors_in_win_shift{i}"] = voxel_info[f"coors_in_win_shift{i}"][
                keep_mask
            ]

            # voxel_num_before_drop_prv = len(voxel_info_prv['voxel_coords'])
            voxel_info_prv[f"voxel_keep_inds_shift{i}"] = voxel_keep_inds_prv[
                keep_mask_prv
            ]
            voxel_info_prv[f"voxel_drop_level_shift{i}"] = drop_lvl_prv[keep_mask_prv]
            voxel_info_prv[f"batch_win_inds_shift{i}"] = batch_win_inds_prv[
                keep_mask_prv
            ]
            voxel_info_prv[f"voxel_features_shift{i}"] = voxel_info_prv[
                "voxel_features"
            ][keep_mask_prv]
            voxel_info_prv[f"voxel_coords_shift{i}"] = voxel_info_prv["voxel_coords"][
                keep_mask_prv
            ]
            voxel_info_prv[f"coors_in_win_shift{i}"] = voxel_info_prv[
                f"coors_in_win_shift{i}"
            ][keep_mask_prv]

        return voxel_info, voxel_info_prv

    def forward(self, input_dict):
        """
        Args:
            input_dict:
                voxel_feats: shape=[N, C], N is the voxel num in the batch.
                coors: shape=[N, 4], [b, z, y, x]
        Returns:
            feat_3d_dict: contains region features (feat_3d) of each region batching level. Shape of feat_3d is [num_windows, num_max_tokens, C].
            flat2win_inds_list: two dict containing transformation information for non-shifted grouping and shifted grouping, respectively. The two dicts are used in function flat2window and window2flat.
            voxel_info: dict containing extra information of each voxel for usage in the backbone.
        """

        def get_voxel_info(input_dict):
            voxel_features = input_dict["voxel_features"]
            voxel_coords = input_dict["voxel_coords"].long()
            voxel_shuffle_inds = input_dict["voxel_shuffle_inds"]
            grid_size = input_dict["grid_size"]

            if self.shuffle_voxels:
                # shuffle the voxels to make the drop process uniform.
                shuffle_inds = torch.randperm(len(voxel_features))
                voxel_features = voxel_features[shuffle_inds]
                voxel_coords = voxel_coords[shuffle_inds]
                voxel_shuffle_inds = voxel_shuffle_inds[shuffle_inds]

            voxel_info = self.window_partition(voxel_coords, grid_size)
            voxel_info["voxel_features"] = voxel_features
            voxel_info["voxel_coords"] = voxel_coords
            voxel_info["voxel_shuffle_inds"] = voxel_shuffle_inds

            return voxel_info

        voxel_info = get_voxel_info(input_dict[0])
        voxel_info_prv = get_voxel_info(input_dict[1])
        voxel_info, voxel_info_prv = self.drop_voxel(
            voxel_info, voxel_info_prv, 2
        )  # voxel_info is
        # updated in this function

        for i in range(2):
            voxel_info[f"flat2win_inds_shift{i}"] = sst_utils.get_flat2win_inds_v2(
                voxel_info[f"batch_win_inds_shift{i}"],
                voxel_info[f"voxel_drop_level_shift{i}"],
                self.drop_info,
            )
            voxel_info_prv[f"flat2win_inds_shift{i}"] = sst_utils.get_flat2win_inds_v2(
                voxel_info_prv[f"batch_win_inds_shift{i}"],
                voxel_info_prv[f"voxel_drop_level_shift{i}"],
                self.drop_info,
            )

            voxel_info[f"pos_dict_shift{i}"] = self.get_pos_embed(
                voxel_info[f"flat2win_inds_shift{i}"],
                voxel_info[f"coors_in_win_shift{i}"],
                voxel_info[f"voxel_features_shift{i}"].size(1),
            )
            voxel_info_prv[f"pos_dict_shift{i}"] = self.get_pos_embed(
                voxel_info_prv[f"flat2win_inds_shift{i}"],
                voxel_info_prv[f"coors_in_win_shift{i}"],
                voxel_info_prv[f"voxel_features_shift{i}"].size(1),
            )

            voxel_info[f"key_mask_shift{i}"] = self.get_key_padding_mask(
                voxel_info[f"flat2win_inds_shift{i}"]
            )
            voxel_info_prv[f"key_mask_shift{i}"] = self.get_key_padding_mask(
                voxel_info_prv[f"flat2win_inds_shift{i}"]
            )
        return voxel_info, voxel_info_prv


class WCABlock(nn.Module):
    # inherit from SSTBlockV1, except for the following changes:
    # 1. change cfg:
    #       1.1 indice_key/name: sst_block -> wca_block
    #       1.2 NUM_BLOCKS: 2 -> 1 (Temporarily)
    # 2. delete self.conv_down
    # 3. SSTInputLayer() -> SSTInputLayer(temporal=True)
    # 4. BasicShiftBlockV2 -> BasicShiftBlock_WCA
    #
    def __init__(self, model_cfg, input_channels, indice_key, **kwargs):
        indice_key.replace("sst_block", "wca_block")  # change - add
        super().__init__()
        self.model_cfg = model_cfg
        encoder_cfg = model_cfg.ENCODER
        d_model = encoder_cfg.D_MODEL
        stride = encoder_cfg.STRIDE
        if model_cfg.get("WCABlock", None) is not None:
            self.WCABlock_Version = model_cfg.WCABlock["Version"]
        else:
            self.WCABlock_Version = 1
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sst_temporal_input_layer = SSTInputLayer_Temporal(model_cfg.PREPROCESS)
        if encoder_cfg.NUM_BLOCKS == 2:
            encoder_cfg.NUM_BLOCKS = 1
            print("Warning: NUM_BLOCKS is set to 1 for WCA block")
        block_list = []
        for i in range(encoder_cfg.NUM_BLOCKS):
            block_list.append(
                BasicShiftBlock_WCA(
                    d_model,
                    encoder_cfg.NHEAD,
                    encoder_cfg.DIM_FEEDFORWARD,
                    encoder_cfg.DROPOUT,
                    encoder_cfg.ACTIVATION,
                    batch_first=False,
                    layer_cfg=encoder_cfg.LAYER_CFG,
                )
            )

        self.encoder_blocks = nn.ModuleList(block_list)
        self.conv_out = post_act_block(
            d_model, d_model, 3, norm_fn=norm_fn, indice_key=f"{indice_key}_subm", dim=2
        )

    def decouple_sp_tensor(self, sp_tensor):
        voxel_features = sp_tensor.features
        voxel_coords = sp_tensor.indices.long()
        voxel_coords = torch.cat(
            [
                voxel_coords[:, 0:1],
                torch.zeros_like(voxel_coords[:, 0:1]),
                voxel_coords[:, 1:],
            ],
            dim=-1,
        )  # (bs_idx, 0, y, x)
        grid_size = sp_tensor.spatial_shape
        grid_size = [grid_size[1], grid_size[0], 1]  # [X, Y, 1]
        return voxel_features, voxel_coords, grid_size

    def _get_preprocess_dict(self, voxel_features, voxel_coords, grid_size):
        voxel_shuffle_inds = torch.arange(
            voxel_coords.shape[0], device=voxel_coords.device, dtype=torch.long
        )
        return {
            "voxel_features": voxel_features,
            "voxel_coords": voxel_coords,
            "voxel_shuffle_inds": voxel_shuffle_inds,
            "grid_size": grid_size,
        }

    def encoder_forward(self, voxel_features, voxel_coords, grid_size, spconv_prv):

        preprocess_dict_list = [
            self._get_preprocess_dict(voxel_features, voxel_coords, grid_size),
            self._get_preprocess_dict(*self.decouple_sp_tensor(spconv_prv)),
        ]
        voxel_info, voxel_info_prv = self.sst_temporal_input_layer(preprocess_dict_list)
        # return these keys for cur and prv:
        #  - voxel_features
        #  - voxel_coords
        #  - voxel_features_shift0
        #  - voxel_features_shift1

        num_shifts = 2
        voxel_features = voxel_info["voxel_features"]
        voxel_coords = voxel_info["voxel_coords"]
        voxel_shuffle_inds = voxel_info["voxel_shuffle_inds"]
        ind_dict_list = [
            voxel_info[f"flat2win_inds_shift{i}"] for i in range(num_shifts)
        ]
        # padding_mask_list = [voxel_info[f'key_mask_shift{i}'] for i in range(num_shifts)]
        pos_embed_list = [voxel_info[f"pos_dict_shift{i}"] for i in range(num_shifts)]
        keep_inds_list = [
            voxel_info[f"voxel_keep_inds_shift{i}"] for i in range(num_shifts)
        ]

        voxel_features_prv = voxel_info_prv["voxel_features"]
        # voxel_coords_prv = voxel_info_prv['voxel_coords']
        # voxel_shuffle_inds_prv = voxel_info_prv['voxel_shuffle_inds']
        ind_dict_list_prv = [
            voxel_info_prv[f"flat2win_inds_shift{i}"] for i in range(num_shifts)
        ]
        padding_mask_list_prv = [
            voxel_info_prv[f"key_mask_shift{i}"] for i in range(num_shifts)
        ]
        pos_embed_list_prv = [
            voxel_info_prv[f"pos_dict_shift{i}"] for i in range(num_shifts)
        ]
        keep_inds_list_prv = [
            voxel_info_prv[f"voxel_keep_inds_shift{i}"] for i in range(num_shifts)
        ]

        voxel_features = self.encoder_blocks[0](
            voxel_features,
            pos_embed_list,
            ind_dict_list,
            keep_inds_list,
            padding_mask_list_prv,
            voxel_features_prv,
            pos_embed_list_prv,
            ind_dict_list_prv,
            keep_inds_list_prv,
        )

        return voxel_features, voxel_coords, voxel_shuffle_inds

    def encoder_forward_self(self, voxel_features, voxel_coords, grid_size):
        voxel_shuffle_inds = torch.arange(
            voxel_coords.shape[0], device=voxel_coords.device, dtype=torch.long
        )

        preprocess_dict = {
            "voxel_features": voxel_features,
            "voxel_coords": voxel_coords,
            "voxel_shuffle_inds": voxel_shuffle_inds,
            "grid_size": grid_size,
        }
        voxel_info = self.sst_input_layer(preprocess_dict)

        num_shifts = 2
        voxel_features = voxel_info["voxel_features"]
        voxel_coords = voxel_info["voxel_coords"]
        voxel_shuffle_inds = voxel_info["voxel_shuffle_inds"]
        ind_dict_list = [
            voxel_info[f"flat2win_inds_shift{i}"] for i in range(num_shifts)
        ]
        padding_mask_list = [
            voxel_info[f"key_mask_shift{i}"] for i in range(num_shifts)
        ]
        pos_embed_list = [voxel_info[f"pos_dict_shift{i}"] for i in range(num_shifts)]

        output = voxel_features
        output = self.encoder_blocks[1](
            output, pos_embed_list, ind_dict_list, padding_mask_list
        )
        voxel_features = output

        return voxel_features, voxel_coords, voxel_shuffle_inds

    def forward(self, sp_tensor, sp_tensor_prev, dtime=0):
        voxel_features, voxel_coords, grid_size = self.decouple_sp_tensor(sp_tensor)
        voxel_features_shuffle, voxel_coords_shuffle, voxel_shuffle_inds = (
            self.encoder_forward(
                voxel_features, voxel_coords, grid_size, sp_tensor_prev
            )
        )

        voxel_features_unshuffle = torch.zeros_like(voxel_features)
        voxel_features_unshuffle[voxel_shuffle_inds] = voxel_features_shuffle.to(
            dtype=voxel_features_unshuffle.dtype
        )
        sp_tensor = replace_feature(
            sp_tensor, voxel_features + voxel_features_unshuffle
        )
        sp_tensor = self.conv_out(sp_tensor)
        return sp_tensor


class SiamWCA(nn.Module):
    def __init__(
        self,
        model_cfg,
        input_channels,
        grid_size,
        voxel_size,
        point_cloud_range,
        **kwargs,
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.sparse_shape = grid_size[[1, 0]]

        self.asymmetric = (
            True
            if self.model_cfg.get("ASYMMETRIC", False)
            and self.model_cfg.ASYMMETRIC["ENABLED"]
            else False
        )

        in_channels = input_channels
        sst_block_list = model_cfg.SST_BLOCK_LIST
        self.sst_blocks = nn.ModuleList()
        for sst_block_cfg in sst_block_list:
            self.sst_blocks.append(
                SSTBlockV1(sst_block_cfg, in_channels, sst_block_cfg.NAME)
            )
            in_channels = sst_block_cfg.ENCODER.D_MODEL

        in_channels = input_channels
        if self.asymmetric:
            if (
                hasattr(self.model_cfg.ASYMMETRIC, "HALF_CHANNELS")
                and self.model_cfg.ASYMMETRIC["HALF_CHANNELS"]
            ):
                self.asymmetric_simsiam = False
                self.sst_blocks_prev = nn.ModuleList()
                for sst_block_cfg in sst_block_list:
                    self.sst_blocks_prev.append(
                        SSTBlockV1(
                            sst_block_cfg,
                            in_channels,
                            sst_block_cfg.NAME + "_prev",
                            half_channels=True,
                        )
                    )
                    in_channels = sst_block_cfg.ENCODER.D_MODEL
            elif (
                hasattr(self.model_cfg.ASYMMETRIC, "SimSiam")
                and self.model_cfg.ASYMMETRIC["SimSiam"]
            ):
                self.asymmetric_simsiam = True

        # Inherited from above 5 lines
        wca_block_list = model_cfg.SST_BLOCK_LIST
        self.wca_blocks = nn.ModuleList()
        for wca_block_cfg in wca_block_list:
            in_channels = wca_block_cfg.ENCODER.D_MODEL
            self.wca_blocks.append(
                WCABlock(wca_block_cfg, in_channels, wca_block_cfg.NAME)
            )

        in_channels = 0
        self.deblocks = nn.ModuleList()
        for src in model_cfg.FEATURES_SOURCE:
            conv_cfg = model_cfg.FUSE_LAYER[src]
            self.deblocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        conv_cfg.NUM_FILTER,
                        conv_cfg.NUM_UPSAMPLE_FILTER,
                        conv_cfg.UPSAMPLE_STRIDE,
                        stride=conv_cfg.UPSAMPLE_STRIDE,
                        bias=False,
                    ),
                    nn.BatchNorm2d(
                        conv_cfg.NUM_UPSAMPLE_FILTER, eps=1e-3, momentum=0.01
                    ),
                    # nn.GroupNorm(conv_cfg.NUM_UPSAMPLE_FILTER // 8, conv_cfg.NUM_UPSAMPLE_FILTER, eps=1e-3),
                    nn.ReLU(inplace=True),
                )
            )
            in_channels += conv_cfg.NUM_UPSAMPLE_FILTER

        self.conv_out = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels // len(self.deblocks), 3, padding=1, bias=False
            ),
            nn.BatchNorm2d(in_channels // len(self.deblocks), eps=1e-3, momentum=0.01),
            # nn.GroupNorm(in_channels // len(self.deblocks) // 8,
            #                in_channels // len(self.deblocks),
            #                eps=1e-3),
            nn.ReLU(inplace=True),
        )
        self.num_point_features = in_channels // len(self.deblocks)

    def sparse_encode(
        self, voxel_features, voxel_coords, batch_size, previous_sstblock=False
    ):
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords[:, [0, 2, 3]]
            .contiguous()
            .int(),  # (bs_idx, y_idx, x_idx)
            spatial_shape=self.sparse_shape,
            batch_size=batch_size,
        )

        x = input_sp_tensor
        x_hidden = []
        if self.asymmetric and previous_sstblock:
            for sst_block in self.sst_blocks_prev:
                x = sst_block(x)
                x_hidden.append(x)
        else:
            for sst_block in self.sst_blocks:
                x = sst_block(x)
                x_hidden.append(x)

        # batch_dict.update({
        #     'encoded_spconv_tensor': x_hidden[-1],
        #     'encoded_spconv_tensor_stride': self.sparse_shape[0] // x_hidden[-1].spatial_shape[0]
        # })

        multi_scale_3d_features, multi_scale_3d_strides = {}, {}
        for i in range(len(x_hidden)):
            multi_scale_3d_features[f"x_conv{i + 1}"] = x_hidden[i]
            multi_scale_3d_strides[f"x_conv{i + 1}"] = 2 ** (i + 1)

        return multi_scale_3d_features, multi_scale_3d_strides

    def sparse_cross_attn(
        self, multi_scale_3d_features, multi_scale_3d_features_prev, dtime=0
    ):
        multi_scale_3d_features_new = {}
        for i, wca_block in enumerate(self.wca_blocks):
            x_prev = multi_scale_3d_features_prev[f"x_conv{i + 1}"]
            x = multi_scale_3d_features[f"x_conv{i + 1}"]
            x = wca_block(x, x_prev, dtime)
            multi_scale_3d_features_new[f"x_conv{i + 1}"] = x

        return multi_scale_3d_features_new

    def dense_conv(self, multi_scale_3d_features, multi_scale_3d_strides):
        spatial_features = []
        spatial_features_stride = []
        for i, src in enumerate(self.model_cfg.FEATURES_SOURCE):
            per_features = multi_scale_3d_features[src].dense()
            B, Y, X = (
                per_features.shape[0],
                per_features.shape[-2],
                per_features.shape[-1],
            )
            spatial_features.append(self.deblocks[i](per_features.view(B, -1, Y, X)))
            spatial_features_stride.append(
                multi_scale_3d_strides[src]
                // self.model_cfg.FUSE_LAYER[src].UPSAMPLE_STRIDE
            )
        spatial_features = self.conv_out(
            torch.cat(spatial_features, dim=1)
        )  # (B, C, Y, X)
        spatial_features_stride = spatial_features_stride[0]

        return spatial_features, spatial_features_stride

    def forward(self, batch_dict):
        assert torch.all(batch_dict["voxel_coords"][:, 1] == 0)
        assert torch.all(batch_dict["voxel_coords_prev"][:, 1] == 0)

        batch_size = batch_dict["batch_size"]

        all_voxel_features_prev, all_voxel_coords_prev = (
            batch_dict["voxel_features_prev"],
            batch_dict["voxel_coords_prev"],
        )
        if self.asymmetric and self.asymmetric_simsiam:
            with torch.no_grad():
                multi_scale_3d_features_prev, multi_scale_3d_strides_prev = (
                    self.sparse_encode(
                        all_voxel_features_prev, all_voxel_coords_prev, batch_size
                    )
                )
        else:
            multi_scale_3d_features_prev, multi_scale_3d_strides_prev = (
                self.sparse_encode(
                    all_voxel_features_prev,
                    all_voxel_coords_prev,
                    batch_size,
                    previous_sstblock=True,
                )
            )

        all_voxel_features, all_voxel_coords = (
            batch_dict["voxel_features"],
            batch_dict["voxel_coords"],
        )
        multi_scale_3d_features, multi_scale_3d_strides = self.sparse_encode(
            all_voxel_features, all_voxel_coords, batch_size
        )

        multi_scale_3d_features = self.sparse_cross_attn(
            multi_scale_3d_features,
            multi_scale_3d_features_prev,
            dtime=batch_dict.get("dt", 0),
        )

        spatial_features, spatial_features_stride = self.dense_conv(
            multi_scale_3d_features, multi_scale_3d_strides
        )
        batch_dict["multi_scale_3d_features"] = multi_scale_3d_features
        batch_dict["multi_scale_3d_strides"] = multi_scale_3d_strides
        batch_dict["spatial_features"] = spatial_features
        batch_dict["spatial_features_stride"] = spatial_features_stride
        return batch_dict


def determine_cur_drop_lvl_by_prev_drop_lvl():
    """
    This is a conceptual explanation for how XXX works.
    """

    # Original version (one frame, some details are omitted)
    # 1. batch_win_inds (shift0, shift1) := window_partition(voxel_coords, grid_size)
    # 2. voxel_drop_level (shift0, shift1) = drop_voxel(batch_win_inds)
    #       voxel_drop_level_shift0 = drop_single_shift(batch_win_inds_shift0)
    #           num_per_voxel_before_drop <-- calculate the number of voxels in each windows
    #           for dl in drop_info: (== for drop_level in [0, 1, 2])
    #               range_mask = (num_per_voxel_before_drop >= lower) & (num_per_voxel_before_drop < upper)
    #               drop_lvl_per_voxel[range_mask] = dl
    #               # by doing this, drop_lvl_per_voxel consists three values, 16, 32, 64, indicating the drop level of each voxel
    #       voxel_drop_level_shift1 = drop_single_shift(batch_win_inds_shift1)
    #           ... (same as above)
    # For temporal version, the naive way is repeat the above process twice. However, it leads to the
    # same windows has two drop_level for two frames. To avoid this, we need to determine the
    # drop_level of current frame by the drop_level of previous frame.

    # Target version (two frame, some details are omitted)
    # Assume the drop_level are [1, 2, 4] voxels -> [0, 1, 2] drop level
    batch_win_inds = torch.tensor(np.array([1, 2, 3, 2]), dtype=torch.int32)
    prv_batch_win_inds = torch.tensor(
        np.array([1, 3, 3, 2, 3, 2, 1, 3]), dtype=torch.int32
    )

    # By manually calculating, we obtaining:
    drop_lvl_per_voxel = torch.tensor(
        np.array([0, 1, 0, 1]), dtype=torch.int32
    )  # window 2 has 2
    # voxels, so the drop level is 1
    prv_drop_lvl_per_voxel = torch.tensor(
        np.array([1, 2, 2, 1, 2, 1, 1, 2]), dtype=torch.int32
    )  #
    # window 3 has 4 voxels, so the drop level is 2
    target_drop_lvl_per_voxel = torch.tensor(np.array([1, 1, 2, 1]), dtype=torch.int32)

    # The algorithm:
    val, inv = torch.unique(prv_batch_win_inds, return_inverse=True)
    pass
