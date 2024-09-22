# OpenPCDet PyTorch Dataloader and Evaluation Tools for Waymo Open Dataset
# Reference https://github.com/open-mmlab/OpenPCDet
# Written by Shaoshuai Shi, Chaoxu Guo
# All Rights Reserved 2019-2020.

import os
import pickle
import copy
import numpy as np
import torch
import multiprocessing
import SharedArray
import torch.distributed as dist
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from ...utils import box_utils, common_utils
from ..dataset import DatasetTemplate
from ..waymo.waymo_utils import (
    convert_prv_frame_to_cur,
    remove_ego_points,
    convert_to_local,
    convert_to_global,
)
from tools.visual_utils.open3d_vis_utils import draw_scenes, draw_scenes_with_2pcs
import pdb


class WaymoTemporalDataset(DatasetTemplate):
    def __init__(
        self, dataset_cfg, class_names, training=True, root_path=None, logger=None
    ):
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training,
            root_path=root_path,
            logger=logger,
        )
        ##############################
        # | WHEN           | self.training  | self.mode  | self.split |
        # | -------------- | ------------- | ---------- | ---------- |
        # | train          | True           | 'train'    | 'train' |
        # | val            | False          | 'test'     | 'val' |
        # | create_info & train | False     | 'test'     | 'train' |
        # | create_info & val | False       | 'test'     | 'val' |
        # | create_info & test | False      | 'test'     | 'test' |
        # | create_info & gt | False        | 'test'    | 'train' |
        # Conclusion
        # * self.mode = 'train' only if self.training = True
        # * self.split depends on pre-define in yaml file or self.set_split() (call by self.create_waymo_infos())
        ##############################

        # An overview
        # 1. __init()__:
        #   1.1. call the super().__init()__, check details in parent class
        #   1.2. initialize the self.params
        #   1.3. read ImageSets/train.txt or ImageSets/val.txt --> self.sample_sequence_list (a list
        #           of ALL sequence names)
        #   1.4. self.include_waymo_data() --> self.infos
        #       1.4.1 For each sequence, read sample-/frame-wise infos based on .pkl files (which obtained from
        #               preprocessing) and records intervals' start and end idx
        #       1.4.2 After go through all seqs, merge infos and intervals
        #       (commented) 1.4.3 sample frames every SAMPLED_INTERVAL frames, e.g. 20% = 1 / every 5 frames
        #   1.5. if use_shared_memory: ....
        # 2. __getitem()__:
        #   2.1. get pc_id, sequence_name etc from self.infos[index]
        #   2.2. get lidar points from shared memory or disk
        #   2.3. get annos
        #   2.4. self.prepare_data(), check details in parent class
        # 3. __collate_fn()__:

        self.data_path = self.root_path / self.dataset_cfg.PROCESSED_DATA_TAG
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]

        self.use_sequence_data = (
            True
            if self.dataset_cfg.get("SEQUENCE_CONFIG", None) is not None
            and self.dataset_cfg.SEQUENCE_CONFIG.get("ENABLED", False)
            else False
        )
        if self.use_sequence_data:
            self.sample_offset = self.dataset_cfg.SEQUENCE_CONFIG.SAMPLE_OFFSET

        self.get_sample_sequence_list()

        self.align_two_frames = self.dataset_cfg.get("ALIGN_TWO_FRAMES", False)
        self.limit_max_number_of_points = False
        if self.training and self.dataset_cfg.get("MAX_NUMBER_OF_POINTS", False):
            self.limit_max_number_of_points = True
            self.max_number_of_points = self.dataset_cfg.MAX_NUMBER_OF_POINTS
        if self.training and self.dataset_cfg.get("MAX_NUMBER_OF_POINTS_BACK", False):
            self.limit_max_number_of_points = True
            self.max_number_of_points_back = self.dataset_cfg.MAX_NUMBER_OF_POINTS_BACK

        self.scan_window = (
            self.dataset_cfg["SCAN_WINDOW"]
            if self.training
            else self.dataset_cfg["SCAN_WINDOW_TST"]
        )
        self.sampling_window = int(np.floor(self.scan_window / 3))
        self.fixed_gap = self.dataset_cfg.get("FIXED_GAP", -1)

        self.infos = []
        self.intervals = []
        self.include_waymo_data(self.mode)

        self.use_shared_memory = (
            self.dataset_cfg.get("USE_SHARED_MEMORY", False) and self.training
        )
        if self.use_shared_memory:
            self.shared_memory_file_limit = self.dataset_cfg.get(
                "SHARED_MEMORY_FILE_LIMIT", 0x7FFFFFFF
            )
            self.load_data_to_shared_memory()

        self.same_input = False
        if hasattr(dataset_cfg, "SAME_INPUT") and dataset_cfg.SAME_INPUT:
            self.same_input = True

    def get_sample_sequence_list(self):
        # 1. get_split_dir
        if (
            self.split == "val"
            or self.split == "test"
            or not hasattr(self.dataset_cfg, "DATA_EFFICIENT_BENCHMARK")
            or self.dataset_cfg.DATA_EFFICIENT_BENCHMARK.get("percentile", 1) == 1
        ):
            split_dir = self.root_path / "ImageSets" / (self.split + ".txt")
        else:
            tmp_percentile = self.dataset_cfg.DATA_EFFICIENT_BENCHMARK["percentile"]
            if tmp_percentile == 0.05:
                tmp_file_name = "waymo_infos_train_r_%.2f_%d_sequence_names" % (
                    self.dataset_cfg.DATA_EFFICIENT_BENCHMARK["percentile"],
                    self.dataset_cfg.DATA_EFFICIENT_BENCHMARK["idx"],
                )
            else:
                tmp_file_name = "waymo_infos_train_r_%.1f_%d_sequence_names" % (
                    self.dataset_cfg.DATA_EFFICIENT_BENCHMARK["percentile"],
                    self.dataset_cfg.DATA_EFFICIENT_BENCHMARK["idx"],
                )
            split_dir = (
                self.root_path
                / "MVJAR_Data_Efficient_Benchmark"
                / "sequence_names"
                / (tmp_file_name + ".txt")
            )

        # 2. read split_dir to get sample_sequence_list
        print("split_dir: ", split_dir)
        with self.client.get_local_path(split_dir) as path:
            self.sample_sequence_list = [x.strip() for x in open(path).readlines()]

    def include_waymo_data(self, mode):
        # This func is called twice, one for pre-processing, one from __init__()
        # but in temporal dataset, we only need to call it during __init__()
        self.logger.info("Loading Temporal Waymo dataset")
        waymo_infos = []
        waymo_intervals = []

        num_skipped_infos = 0
        # for k in range(len(self.sample_sequence_list[:2])):  # if debug
        for k in range(len(self.sample_sequence_list)):
            sequence_name = os.path.splitext(self.sample_sequence_list[k])[0]
            info_path = self.data_path / sequence_name / ("%s.pkl" % sequence_name)
            info_path = self.check_sequence_name_with_all_version(info_path)
            if not self.client.exists(info_path):
                num_skipped_infos += 1
                continue
            infos = self.client.load_pickle(info_path)

            seq_start_id = len(waymo_infos)
            seq_end_id = seq_start_id + len(infos)
            intervals = []
            if self.training and self.scan_window > 3:
                for first_ix in range(seq_start_id, seq_end_id, self.sampling_window):
                    last_ix = min(first_ix + self.scan_window, seq_end_id)
                    intervals.append([first_ix, last_ix])
                    if last_ix == seq_end_id:
                        break
            elif self.training and self.scan_window == 2:
                # for training
                # intervals.extend([[first_ix, min(first_ix+self.scan_window, seq_end_id)]
                #                   for first_ix in range(seq_start_id, seq_end_id-1, 2)])
                # [0, 2], [1, 3], [2, 4], [3, 5], ..., [n-2, n], where n == seq_end_id
                # Only the first frame is not used for training
                intervals.append([seq_start_id, seq_start_id + 1])
                intervals.extend(
                    [
                        [first_ix, first_ix + 2]
                        for first_ix in range(seq_start_id, seq_end_id - 1)
                    ]
                )
            else:
                # for and evaluation
                intervals.append([seq_start_id, seq_start_id + 1])
                intervals.extend(
                    [
                        [first_ix, first_ix + 2]
                        for first_ix in range(seq_start_id, seq_end_id - 1)
                    ]
                )

            waymo_infos.extend(infos)
            waymo_intervals.extend(intervals)

        self.intervals.extend(waymo_intervals[:])
        self.logger.info("Total skipped info %s" % num_skipped_infos)
        self.logger.info(
            "Total samples for Temporal Waymo dataset: %d" % (len(waymo_intervals))
        )

        if self.dataset_cfg.SAMPLED_INTERVAL[mode] > 1:
            sampled_waymo_intervals = []
            for k in range(
                0, len(self.intervals), self.dataset_cfg.SAMPLED_INTERVAL[mode]
            ):
                sampled_waymo_intervals.append(self.intervals[k])
            self.intervals = sampled_waymo_intervals
            self.logger.info(
                "Total sampled samples for Waymo dataset: %d" % len(self.intervals)
            )

        # Delete unused infos to save memory
        if self.training:
            new_infos = {}
            for id0, id1 in self.intervals:
                for k in range(id0, id1):
                    if new_infos.get(k, None) is None:
                        new_infos[k] = waymo_infos[k]

            self.infos = new_infos
        else:
            self.infos = waymo_infos[:]

    def load_data_to_shared_memory(self):
        self.logger.info(
            f"Loading training data to shared memory (file limit={self.shared_memory_file_limit})"
        )

        cur_rank, num_gpus = common_utils.get_dist_info()
        all_infos = (
            self.infos[: self.shared_memory_file_limit]
            if self.shared_memory_file_limit < len(self.infos)
            else self.infos
        )
        cur_infos = all_infos[cur_rank::num_gpus]
        for info in cur_infos:
            pc_info = info["point_cloud"]
            sequence_name = pc_info["lidar_sequence"]
            sample_idx = pc_info["sample_idx"]

            sa_key = f"{sequence_name}___{sample_idx}"
            if os.path.exists(f"/dev/shm/{sa_key}"):
                continue

            points = self.get_lidar(sequence_name, sample_idx)
            common_utils.sa_create(f"shm://{sa_key}", points)

        dist.barrier()
        self.logger.info("Training data has been saved to shared memory")

    def clean_shared_memory(self):
        self.logger.info(
            f"Clean training data from shared memory (file limit={self.shared_memory_file_limit})"
        )

        cur_rank, num_gpus = common_utils.get_dist_info()
        all_infos = (
            self.infos[: self.shared_memory_file_limit]
            if self.shared_memory_file_limit < len(self.infos)
            else self.infos
        )
        cur_infos = all_infos[cur_rank::num_gpus]
        for info in cur_infos:
            pc_info = info["point_cloud"]
            sequence_name = pc_info["lidar_sequence"]
            sample_idx = pc_info["sample_idx"]

            sa_key = f"{sequence_name}___{sample_idx}"
            if not os.path.exists(f"/dev/shm/{sa_key}"):
                continue

            SharedArray.delete(f"shm://{sa_key}")

        if num_gpus > 1:
            dist.barrier()
        self.logger.info("Training data has been deleted from shared memory")

    def check_sequence_name_with_all_version(self, sequence_file):
        if not self.client.exists(sequence_file):
            found_sequence_file = sequence_file
            for pre_text in ["training", "validation", "testing"]:
                if not self.client.exists(sequence_file):
                    temp_sequence_file = Path(
                        str(sequence_file).replace("segment", pre_text + "_segment")
                    )
                    if self.client.exists(temp_sequence_file):
                        found_sequence_file = temp_sequence_file
                        break
            if not self.client.exists(found_sequence_file):
                found_sequence_file = Path(
                    str(sequence_file).replace("_with_camera_labels", "")
                )
            if self.client.exists(found_sequence_file):
                sequence_file = found_sequence_file
        return sequence_file

    def get_infos(
        self,
        raw_data_path,
        save_path,
        num_workers=multiprocessing.cpu_count(),
        has_label=True,
        sampled_interval=1,
    ):
        from functools import partial
        from . import waymo_utils

        print(
            "---------------The waymo sample interval is %d, total sequecnes is %d-----------------"
            % (sampled_interval, len(self.sample_sequence_list))
        )

        process_single_sequence = partial(
            waymo_utils.process_single_sequence,
            client=self.client,
            save_path=save_path,
            sampled_interval=sampled_interval,
            has_label=has_label,
        )
        sample_sequence_file_list = [
            self.check_sequence_name_with_all_version(raw_data_path / sequence_file)
            for sequence_file in self.sample_sequence_list
        ]

        with multiprocessing.Pool(num_workers) as p:
            sequence_infos = list(
                tqdm(
                    p.imap(process_single_sequence, sample_sequence_file_list),
                    total=len(sample_sequence_file_list),
                )
            )

        all_sequences_infos = [item for infos in sequence_infos for item in infos]
        return all_sequences_infos

    def get_lidar(self, sequence_name, sample_idx):
        lidar_file = self.data_path / sequence_name / ("%04d.npy" % sample_idx)
        point_features = self.client.load_npy(
            lidar_file
        )  # (N, 7): [x, y, z, intensity, elongation, NLZ_flag]

        points_all, NLZ_flag = point_features[:, 0:5], point_features[:, 5]
        if not self.dataset_cfg.get("DISABLE_NLZ_FLAG_ON_POINTS", False):
            points_all = points_all[NLZ_flag == -1]
        points_all[:, 3] = np.tanh(points_all[:, 3])
        return points_all

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.intervals) * self.total_epochs

        return len(self.intervals)

    def _combine_two_pcs(self, pc0, pc1):
        pc0 = np.hstack((pc0, np.zeros((pc0.shape[0], 1))))
        pc1 = np.hstack((pc1, np.ones((pc1.shape[0], 1))))
        points = np.vstack((pc0, pc1))
        return points

    def _split_two_pcs(self, points):
        pc0 = points[points[:, -1] == 0, :-1]
        pc1 = points[points[:, -1] == 1, :-1]
        return pc0, pc1

    def _combine_two_pcs_with_delimiter(self, pc0, pc1, delimiter=-np.inf):
        p_delimiter = np.repeat(delimiter, pc0.shape[1])
        return np.vstack((p_delimiter, pc0, p_delimiter, pc1, p_delimiter))

    def _attach_group_ids(self, points):
        pcd_parse_idx = np.unique(np.argwhere(points[:, -1] == -np.inf))

        points = np.hstack((points, np.zeros((points.shape[0], 1))))
        added_pts = points[: pcd_parse_idx[0]]
        points_prev = points[pcd_parse_idx[0] + 1 : pcd_parse_idx[1]]
        points_cur = points[pcd_parse_idx[1] + 1 : pcd_parse_idx[2]]

        points_cur[:, -1] = 1
        res_points = np.vstack((points_prev, points_cur))
        if len(added_pts) != 0:
            res_points = np.vstack((added_pts, res_points))
            # copy added points to current frame
            added_pts[:, -1] = 1
            res_points = np.vstack((added_pts, res_points))
        return res_points

    def _limite_number_of_points(self, points, max_points):
        if points.shape[0] > max_points:
            points = points[
                np.random.choice(points.shape[0], max_points, replace=False)
            ]
        return points

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.intervals)

        num_frames = self.intervals[index][1] - self.intervals[index][0]
        if self.dataset_cfg.get("RAND_PREV", False):
            idx = self.intervals[index][1] - 1
            if num_frames <= 2:
                idx_prev = self.intervals[index][0]
            else:
                if self.fixed_gap >= 2:
                    idx_prev = max(self.intervals[index][0], idx - self.fixed_gap)
                else:
                    if self.minimum_gap is not None:
                        idx_prev = np.random.choice(
                            np.arange(
                                self.intervals[index][0], idx - self.minimum_gap + 1
                            ),
                            1,
                        )[0]
                    else:
                        idx_prev = np.random.choice(
                            np.arange(self.intervals[index][0], idx), 1
                        )[0]
            # print("idx_prev: %d, idx: %d, prev_sample_idx: %d, sample_idx: %d" % (idx_prev, idx,
            #     self.infos[idx_prev]['point_cloud']['sample_idx'], self.infos[idx]['point_cloud']['sample_idx']))
        else:
            # if it is a normal sample (sample with self.scan_window scans) sample one pcd from begining and one from end
            if self.training and self.scan_window > 3:
                if num_frames == self.scan_window:
                    t_frames = np.random.choice(
                        np.arange(self.sampling_window), 2, replace=True
                    )
                    t_frames[1] += 2 * self.sampling_window
                else:
                    # if it is the last sample from the seq (less samples than self.scan_window) sample two random scans (w/o begin-end restriction)
                    t_frames = np.random.choice(np.arange(num_frames), 2, replace=False)
            else:
                # for training and evaluation
                # if num_frames == 0 or num_frames == 1:
                # t_frames = np.array([0, num_frames])
                if num_frames == 1:
                    # for evaluation
                    t_frames = np.array([0, 0])
                else:
                    # for training/FT
                    # t_frames = np.random.choice(np.arange(num_frames), 2, replace=False)
                    t_frames = np.array([0, 1])
            idx_prev = self.intervals[index][0] + min(t_frames)
            idx = self.intervals[index][0] + max(t_frames)
            assert idx < self.intervals[index][1]
        # print("{:3d} === {:3d} === gap= {:3d}".format(
        #     self.intervals[index][0], self.intervals[index][1]-1, idx_prev-idx))

        info = copy.deepcopy(self.infos[idx])
        pc_info = info["point_cloud"]
        sequence_name = pc_info["lidar_sequence"]
        sample_idx = pc_info["sample_idx"]
        info["frame_id"] = sequence_name + ("_%03d" % sample_idx)

        if hasattr(self, "same_input") and self.same_input:
            idx_prev = idx
        info_prev = copy.deepcopy(self.infos[idx_prev])
        sample_idx_prev = info_prev["point_cloud"]["sample_idx"]

        if self.use_shared_memory and index < self.shared_memory_file_limit:
            sa_key = f"{sequence_name}___{sample_idx}"
            points = SharedArray.attach(f"shm://{sa_key}").copy()
        else:
            points_prev = self.get_lidar(sequence_name, sample_idx_prev)
            if self.same_input:
                points_prev[:, :3] += np.random.normal(
                    0, 0.01, [points_prev.shape[0], 3]
                )
            points = self.get_lidar(sequence_name, sample_idx)

        if self.align_two_frames:
            pose_cur = info["pose"]
            pose_prev = info_prev["pose"]
            points_prev = convert_prv_frame_to_cur(points_prev, pose_prev, pose_cur)
            points_prev = remove_ego_points(points_prev)

        if self.limit_max_number_of_points and hasattr(self, "max_number_of_points"):
            points = self._limite_number_of_points(points, self.max_number_of_points)
            points_prev = self._limite_number_of_points(
                points_prev, self.max_number_of_points
            )

        if self.use_sequence_data:
            if self.align_two_frames and self.sample_offset[0] == -1:
                points = np.vstack((points_prev, points))
            else:
                # TBU: add timestamps https://github.com/open-mmlab/OpenPCDet/blob/255db8f02a8bd07211d2c91f54602d63c4c93356/pcdet/datasets/waymo/waymo_dataset.py#L280
                # TBU: add pred_boxes https://github.com/open-mmlab/OpenPCDet/blob/255db8f02a8bd07211d2c91f54602d63c4c93356/pcdet/datasets/waymo/waymo_dataset.py#L291C20-L291C20

                points_global = np.empty((0, points.shape[1]), dtype=points.dtype)
                for offset in range(self.sample_offset[0], self.sample_offset[1]):
                    sample_idx = pc_info["sample_idx"] + offset
                    if sample_idx < 0:
                        continue
                    points_prev = self.get_lidar(sequence_name, sample_idx)
                    pose_prev = self.infos[index + offset]["pose"]
                    points_prv2global = convert_to_global(points_prev, pose_prev)
                    points_global = np.vstack((points_global, points_prv2global))

                if points_global.shape[0] > 0:
                    points_prev_all = convert_to_local(points_global, info["pose"])
                    points_prev_all = remove_ego_points(points_prev_all, 1.0)
                    points = np.vstack((points_prev_all, points))
                    # print("Prev: ", points_prev_all.shape, "Merged: ", points.shape)

        # merged_points = self._combine_two_pcs(points_prev, points)

        input_dict = {
            "points": points,
            "frame_id": info["frame_id"],
        }

        if "annos" in info:
            annos = info["annos"]
            annos = common_utils.drop_info_with_name(annos, name="unknown")

            if self.dataset_cfg.get("INFO_WITH_FAKELIDAR", False):
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(
                    annos["gt_boxes_lidar"]
                )
            else:
                gt_boxes_lidar = annos["gt_boxes_lidar"]

            if self.training and self.dataset_cfg.get(
                "FILTER_EMPTY_BOXES_FOR_TRAIN", False
            ):
                # annos['num_points_in_gt'] comes from original waymo data
                mask = annos["num_points_in_gt"] > 0  # filter empty boxes
                annos["name"] = annos["name"][mask]
                gt_boxes_lidar = gt_boxes_lidar[mask]
                annos["num_points_in_gt"] = annos["num_points_in_gt"][mask]

            input_dict.update(
                {
                    "gt_names": annos["name"],
                    "gt_boxes": gt_boxes_lidar,
                    "num_points_in_gt": annos.get("num_points_in_gt", None),
                }
            )

        data_dict = self.prepare_data(data_dict=input_dict, points_prev=points_prev)
        if data_dict is None:
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)
        else:
            data_dict["metadata"] = info.get("metadata", info["frame_id"])
            data_dict.pop("num_points_in_gt", None)

            points_prev, points = self._split_two_pcs(data_dict["points"])
            if self.limit_max_number_of_points and hasattr(
                self, "max_number_of_points_back"
            ):
                points = self._limite_number_of_points(
                    points, self.max_number_of_points_back
                )
                points_prev = self._limite_number_of_points(
                    points_prev, self.max_number_of_points_back
                )

            data_dict["points_prev"] = points_prev
            data_dict["points"] = points

            # draw_scenes_with_2pcs(points, points_prev, data_dict['gt_boxes'])

            data_dict["dt"] = np.array(idx - idx_prev)
            return data_dict

    def prepare_data(self, data_dict, points_prev):
        """
        Args:
            data_dict:
                points: optional, (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:
            data_dict_for_augmentation = {
                **data_dict,
                "cur_epoch": self.cur_epoch,
                "total_epochs": self.total_epochs,
            }

            # stack two points before data augmentation
            data_dict_for_augmentation["points"] = self._combine_two_pcs_with_delimiter(
                points_prev, data_dict["points"], delimiter=-np.inf
            )

            if data_dict.get("gt_boxes", None) is not None:
                gt_boxes_mask = np.array(
                    [n in self.class_names for n in data_dict["gt_names"]],
                    dtype=np.bool_,
                )
                data_dict_for_augmentation.update({"gt_boxes_mask": gt_boxes_mask})

            data_dict = self.data_augmentor.forward(
                data_dict=data_dict_for_augmentation
            )

            data_dict["points"] = self._attach_group_ids(data_dict["points"])
        else:
            data_dict["points"] = self._combine_two_pcs(
                points_prev, data_dict["points"]
            )

        if data_dict.get("gt_boxes", None) is not None:
            selected = common_utils.keep_arrays_by_name(
                data_dict["gt_names"], self.class_names
            )
            data_dict["gt_boxes"] = data_dict["gt_boxes"][selected]
            data_dict["gt_names"] = data_dict["gt_names"][selected]
            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in data_dict["gt_names"]],
                dtype=np.int32,
            )
            gt_boxes = np.concatenate(
                (data_dict["gt_boxes"], gt_classes.reshape(-1, 1).astype(np.float32)),
                axis=1,
            )
            data_dict["gt_boxes"] = gt_boxes

            if data_dict.get("gt_boxes2d", None) is not None:
                data_dict["gt_boxes2d"] = data_dict["gt_boxes2d"][selected]
                gt_boxes2d = np.concatenate(
                    (
                        data_dict["gt_boxes2d"],
                        gt_classes.reshape(-1, 1).astype(np.float32),
                    ),
                    axis=1,
                )
                data_dict["gt_boxes2d"] = gt_boxes2d

        if data_dict.get("points", None) is not None:
            data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(data_dict=data_dict)

        if (
            self.training
            and data_dict.get("gt_boxes", None) is not None
            and len(data_dict["gt_boxes"]) == 0
        ):
            return None  # override parent's __getitem__ and only change this row

        # Delete these keys if exist
        data_dict.pop("gt_names", None)
        data_dict.pop("cur_epoch", None)
        data_dict.pop("total_epochs", None)

        return data_dict

    @staticmethod
    def generate_prediction_dicts(
        batch_dict, pred_dicts, class_names, output_path=None
    ):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """

        def get_template_prediction(num_samples):
            ret_dict = {
                "name": np.zeros(num_samples),
                "score": np.zeros(num_samples),
                "boxes_lidar": np.zeros([num_samples, 7]),
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict["pred_scores"].cpu().numpy()
            pred_boxes = box_dict["pred_boxes"].cpu().numpy()
            pred_labels = box_dict["pred_labels"].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict["name"] = np.array(class_names)[pred_labels - 1]
            pred_dict["score"] = pred_scores
            pred_dict["boxes_lidar"] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict["frame_id"] = batch_dict["frame_id"][index]
            single_pred_dict["metadata"] = batch_dict["metadata"][index]
            annos.append(single_pred_dict)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):

        def kitti_eval(eval_det_annos, eval_gt_annos):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            map_name_to_kitti = {
                "Vehicle": "Car",
                "Pedestrian": "Pedestrian",
                "Cyclist": "Cyclist",
                "Sign": "Sign",
                "Car": "Car",
            }
            kitti_utils.transform_annotations_to_kitti_format(
                eval_det_annos, map_name_to_kitti=map_name_to_kitti
            )
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos,
                map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get("INFO_WITH_FAKELIDAR", False),
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos,
                dt_annos=eval_det_annos,
                current_classes=kitti_class_names,
            )
            return ap_result_str, ap_dict

        def waymo_eval(eval_det_annos, eval_gt_annos):
            from .waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator

            eval = OpenPCDetWaymoDetectionMetricsEstimator()

            ap_dict = eval.waymo_evaluation(
                eval_det_annos,
                eval_gt_annos,
                class_name=class_names,
                distance_thresh=1000,
                fake_gt_infos=self.dataset_cfg.get("INFO_WITH_FAKELIDAR", False),
            )
            ap_result_str = "\n"
            for key in ap_dict:
                ap_dict[key] = ap_dict[key][0]
                ap_result_str += "%s: %.4f \n" % (key, ap_dict[key])

            return ap_result_str, ap_dict

        def waymo_custom_eval(eval_det_annos, infos, output_path):
            from . import waymo_utils

            waymo_utils.create_pd_detection(eval_det_annos, infos, output_path)
            return "", {}

        eval_det_annos = copy.deepcopy(det_annos)

        if kwargs["eval_metric"] in ["kitti", "waymo"]:
            if "annos" not in self.infos[0].keys():
                return "No ground-truth boxes for evaluation", {}
            eval_gt_annos = [copy.deepcopy(info["annos"]) for info in self.infos]
            if kwargs["eval_metric"] == "kitti":
                ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos)
            else:
                ap_result_str, ap_dict = waymo_eval(eval_det_annos, eval_gt_annos)
        elif kwargs["eval_metric"] == "waymo_custom":
            ap_result_str, ap_dict = waymo_custom_eval(
                eval_det_annos, self.infos, kwargs["output_path"]
            )
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict


def create_waymo_infos(
    dataset_cfg,
    class_names,
    data_path,
    save_path,
    raw_data_tag="raw_data",
    processed_data_tag="waymo_processed_data",
    workers=min(32, multiprocessing.cpu_count()),
):
    raise NotImplementedError


def vis_samples(
    dataset_cfg, class_names, data_path, processed_data_tag="waymo_processed_data"
):
    from torch.utils.data import DataLoader
    from tools.visual_utils.visualize_utils import draw_scenes

    dataset = WaymoTemporalDataset(
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=data_path,
        training=True,
        logger=common_utils.create_logger(),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        pin_memory=True,
        num_workers=2,
        shuffle=False,
        collate_fn=dataset.collate_batch,
        drop_last=True,
        sampler=None,
        timeout=0,
    )

    for batch_dict in dataloader:
        print(batch_dict.keys())
        print(batch_dict["frame_id"])
        print(batch_dict["metadata"])
        print(batch_dict["points"].shape)
        print(batch_dict["voxels"].shape)
        print(batch_dict["gt_boxes"].shape)

        # for
        points = batch_dict["points"][:, 1:].numpy()
        gt_boxes = batch_dict["gt_boxes_lidar"][:, 1:].numpy()

        draw_scenes(
            points,
        )


if __name__ == "__main__":
    # test_func = 'vis_samples'
    test_func = "voxel_gen"

    if test_func == "vis_samples":
        cfg_file = "tools/cfgs/dataset_configs/waymo_temporal_dataset.yaml"
        processed_data_tag = "waymo_processed_data"

        import yaml
        from easydict import EasyDict

        try:
            yaml_config = yaml.safe_load(open(cfg_file), Loader=yaml.FullLoader)
        except:
            yaml_config = yaml.safe_load(open(cfg_file))
        dataset_cfg = EasyDict(yaml_config)
        ROOT_DIR = (Path(__file__).resolve().parent / "../../../").resolve()
        dataset_cfg.PROCESSED_DATA_TAG = processed_data_tag

        vis_samples(
            dataset_cfg=dataset_cfg,
            class_names=["Vehicle", "Pedestrian", "Cyclist"],
            data_path=ROOT_DIR / "data" / "waymo",
            processed_data_tag=processed_data_tag,
        )
    elif test_func == "voxel_gen":
        # Test the VoxelGenerator used by waymo_dataset.py
        from pcdet.datasets.processor.data_processor import VoxelGeneratorWrapper

        np.random.seed(50051)

        # parameters from waymo_temporal_models/gd_mae_ssl.yaml
        gen = VoxelGeneratorWrapper(
            vsize_xyz=[0.32, 0.32, 6.0],
            coors_range_xyz=[-74.88, -74.88, -2.0, 74.88, 74.88, 4.0],
            num_point_features=3,
            max_num_points_per_voxel=5,
            max_num_voxels=5000,
        )
        # voxel size: 468 x 468 x 1

        pc = np.random.uniform(-3.2, 3.2, size=[1000, 3])
        pc[0, :] = np.array([-74.68, -74.68, 0.0])
        voxels, indices, num_p_in_vx = gen.generate(pc)
        # shapes of outputs:
        # voxels: [#voxels, max_num_points_per_voxel, #point_features]
        # indices: [#voxels, 3 (voxel coords)]
        # num_p_in_vx: [#voxels, ], <= max_num_points_per_voxel

        print(f"------Raw Voxels {voxels.shape[0]}-------")
        print(voxels[0])
        pass

        # Conclusion:
        # 1. The voxel generator is range-based, which means the [-74.88, -74.88, x] is the origin in
        # voxel coordinates.
