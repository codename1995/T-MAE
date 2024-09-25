import copy
import numpy as np
from pathlib import Path

from ..dataset import DatasetTemplate
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from .once_eval.once_utils import convert_prv_frame_to_cur, remove_ego_points
from tools.visual_utils.open3d_vis_utils import draw_scenes, draw_scenes_with_2pcs
import pdb


class ONCETemporalDataset(DatasetTemplate):
    def __init__(
        self, dataset_cfg, class_names, training=True, root_path=None, logger=None
    ):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training,
            root_path=root_path,
            logger=logger,
        )
        self.split = (
            dataset_cfg.DATA_SPLIT["train"]
            if training
            else dataset_cfg.DATA_SPLIT["test"]
        )
        assert self.split in [
            "train",
            "val",
            "test",
            "raw_small",
            "raw_medium",
            "raw_large",
        ]

        split_dir = self.root_path / "ImageSets" / (self.split + ".txt")
        with self.client.get_local_path(split_dir) as path:
            self.sample_seq_list = [x.strip() for x in open(path).readlines()]

        self.cam_names = ["cam01", "cam03", "cam05", "cam06", "cam07", "cam08", "cam09"]
        self.cam_tags = [
            "top",
            "top2",
            "left_back",
            "left_front",
            "right_front",
            "right_back",
            "back",
        ]

        self.align_two_frames = self.dataset_cfg.get("ALIGN_TWO_FRAMES", False)
        self.scan_window = self.dataset_cfg.get(
            "SCAN_WINDOW", 1
        )  # 1 means duplicate input
        self.sampling_window = int(np.floor(self.scan_window / 3))
        self.fixed_gap = self.dataset_cfg.get("FIXED_GAP", -1)  # -1 means invalid

        self.once_infos = []
        self.once_intervals = []
        self.include_once_data(self.split)

    def include_once_data(self, split):
        if self.logger is not None:
            self.logger.info("Loading ONCE dataset")
        once_infos = []
        once_intervals = []

        for info_path in self.dataset_cfg.INFO_PATH[split]:
            info_path = self.root_path / info_path
            if not self.client.exists(info_path):
                continue
            infos = self.client.load_pickle(info_path)
            once_infos.extend(infos)

        seq_id = ""
        start_id = 0
        for i, info in enumerate(once_infos):
            # for i, info in enumerate(once_infos[:5000]):  # if debug
            if seq_id != info["sequence_id"] or i == len(once_infos) - 1:
                seq_id = info["sequence_id"]
                intervals = self._generate_intervals(start_id, i, self.scan_window)
                start_id = i

                once_intervals.extend(intervals)

        def check_annos(once_interval):
            return "annos" in once_infos[once_interval[1] - 1]

        if self.split in ["train", "val"]:
            once_intervals = list(filter(check_annos, once_intervals))

        self.once_infos.extend(once_infos)
        self.once_intervals.extend(once_intervals)

        if self.logger is not None:
            self.logger.info(
                "Total samples for ONCE dataset: %d" % (len(once_intervals))
            )

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg,
            class_names=self.class_names,
            training=self.training,
            root_path=self.root_path,
            logger=self.logger,
        )
        self.split = split
        split_dir = self.root_path / "ImageSets" / (self.split + ".txt")
        with self.client.get_local_path(split_dir) as path:
            self.sample_seq_list = [x.strip() for x in open(path).readlines()]
        self.once_infos = []
        self.include_once_data(self.split)

    def get_lidar(self, sequence_id, frame_id):
        lidar_file = (
            self.root_path / "data" / sequence_id / "lidar_roof" / ("%s.bin" % frame_id)
        )
        return self.client.load_to_numpy(str(lidar_file), dtype=np.float32).reshape(
            -1, 4
        )

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.once_intervals) * self.total_epochs

        return len(self.once_intervals)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.once_intervals)

        num_frames = self.once_intervals[index][1] - self.once_intervals[index][0]
        idx = self.once_intervals[index][1] - 1
        if self.fixed_gap == -1:
            if num_frames == 1:
                idx_prev = idx
            else:
                start_idx = self.once_intervals[index][0]
                idx_prev = np.random.choice(
                    np.arange(start_idx, start_idx + self.sampling_window), 1
                )[0]
        else:
            idx_prev = max(self.once_intervals[index][0], idx - self.fixed_gap)
        assert idx_prev <= idx
        assert idx < self.once_intervals[index][1]

        info = copy.deepcopy(self.once_infos[idx])
        frame_id = info["frame_id"]
        seq_id = info["sequence_id"]
        points = self.get_lidar(seq_id, frame_id)

        info_prev = copy.deepcopy(self.once_infos[idx_prev])
        frame_id_prev = info_prev["frame_id"]
        points_prev = self.get_lidar(seq_id, frame_id_prev)

        points = remove_ego_points(points, 2)
        points_prev = remove_ego_points(points_prev, 2)
        if self.align_two_frames and frame_id != frame_id_prev:
            pose_cur = info["pose"]
            pose_prev = info_prev["pose"]
            try:
                points_prev = convert_prv_frame_to_cur(points_prev, pose_prev, pose_cur)
            except ValueError:
                print("ValueError, probably Found zero norm quaternions in `quat`")
                print(
                    "seq_id: %s, frame_id: %s, frame_id_prev: %s"
                    % (seq_id, frame_id, frame_id_prev)
                )
                print("pose_cur: ", pose_cur)
                print("pose_prev: ", pose_prev)
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)

        input_dict = {
            "points": points,
            "frame_id": frame_id,
        }

        if "annos" in info:
            annos = info["annos"]
            input_dict.update(
                {
                    "gt_names": annos["name"],
                    "gt_boxes": annos["boxes_3d"],
                    "num_points_in_gt": annos.get("num_points_in_gt", None),
                }
            )

        data_dict = self.prepare_data(data_dict=input_dict, points_prev=points_prev)
        if data_dict is None:  # no gt boxes
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)
        else:
            data_dict.pop("num_points_in_gt", None)

            points_prev, points = self._split_two_pcs(data_dict["points"])

            data_dict["points_prev"] = points_prev
            data_dict["points"] = points

            return data_dict

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

    def prepare_data(self, data_dict, points_prev=None):
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
            return None

        # Delete these keys if exist
        data_dict.pop("gt_names", None)
        data_dict.pop("cur_epoch", None)
        data_dict.pop("total_epochs", None)

        return data_dict

    def get_infos(self, num_workers=4, sample_seq_list=None):
        import concurrent.futures as futures
        import json

        root_path = self.root_path
        cam_names = self.cam_names

        """
        # dataset json format
        {
            'meta_info': 
            'calib': {
                'cam01': {
                    'cam_to_velo': list
                    'cam_intrinsic': list
                    'distortion': list
                }
                ...
            }
            'frames': [
                {
                    'frame_id': timestamp,
                    'annos': {
                        'names': list
                        'boxes_3d': list of list
                        'boxes_2d': {
                            'cam01': list of list
                            ...
                        }
                    }
                    'pose': list
                },
                ...
            ]
        }
        # open pcdet format
        {
            'meta_info':
            'sequence_id': seq_idx
            'frame_id': timestamp
            'timestamp': timestamp
            'lidar': path
            'cam01': path
            ...
            'calib': {
                'cam01': {
                    'cam_to_velo': np.array
                    'cam_intrinsic': np.array
                    'distortion': np.array
                }
                ...
            }
            'pose': np.array
            'annos': {
                'name': np.array
                'boxes_3d': np.array
                'boxes_2d': {
                    'cam01': np.array
                    ....
                }
            }          
        }
        """

        def process_single_sequence(seq_idx):
            print("%s seq_idx: %s" % (self.split, seq_idx))
            seq_infos = []
            seq_path = Path(root_path) / "data" / seq_idx
            json_path = seq_path / ("%s.json" % seq_idx)
            info_this_seq = self.client.load_json(json_path)
            meta_info = info_this_seq["meta_info"]
            calib = info_this_seq["calib"]
            for f_idx, frame in enumerate(info_this_seq["frames"]):
                frame_id = frame["frame_id"]
                if f_idx == 0:
                    prev_id = None
                else:
                    prev_id = info_this_seq["frames"][f_idx - 1]["frame_id"]
                if f_idx == len(info_this_seq["frames"]) - 1:
                    next_id = None
                else:
                    next_id = info_this_seq["frames"][f_idx + 1]["frame_id"]
                pc_path = str(seq_path / "lidar_roof" / ("%s.bin" % frame_id))
                pose = np.array(frame["pose"])
                frame_dict = {
                    "sequence_id": seq_idx,
                    "frame_id": frame_id,
                    "timestamp": int(frame_id),
                    "prev_id": prev_id,
                    "next_id": next_id,
                    "meta_info": meta_info,
                    "lidar": pc_path,
                    "pose": pose,
                }
                calib_dict = {}
                for cam_name in cam_names:
                    cam_path = str(seq_path / cam_name / ("%s.jpg" % frame_id))
                    frame_dict.update({cam_name: cam_path})
                    calib_dict[cam_name] = {}
                    calib_dict[cam_name]["cam_to_velo"] = np.array(
                        calib[cam_name]["cam_to_velo"]
                    )
                    calib_dict[cam_name]["cam_intrinsic"] = np.array(
                        calib[cam_name]["cam_intrinsic"]
                    )
                    calib_dict[cam_name]["distortion"] = np.array(
                        calib[cam_name]["distortion"]
                    )
                frame_dict.update({"calib": calib_dict})

                if "annos" in frame:
                    annos = frame["annos"]
                    boxes_3d = np.array(annos["boxes_3d"])
                    if boxes_3d.shape[0] == 0:
                        print(frame_id)
                        continue
                    boxes_2d_dict = {}
                    for cam_name in cam_names:
                        boxes_2d_dict[cam_name] = np.array(annos["boxes_2d"][cam_name])
                    annos_dict = {
                        "name": np.array(annos["names"]),
                        "boxes_3d": boxes_3d,
                        "boxes_2d": boxes_2d_dict,
                    }

                    points = self.get_lidar(seq_idx, frame_id)
                    corners_lidar = box_utils.boxes_to_corners_3d(
                        np.array(annos["boxes_3d"])
                    )
                    num_gt = boxes_3d.shape[0]
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)
                    for k in range(num_gt):
                        flag = box_utils.in_hull(points[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annos_dict["num_points_in_gt"] = num_points_in_gt

                    frame_dict.update({"annos": annos_dict})
                seq_infos.append(frame_dict)
            return seq_infos

        sample_seq_list = (
            sample_seq_list if sample_seq_list is not None else self.sample_seq_list
        )
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_sequence, sample_seq_list)
        all_infos = []
        for info in infos:
            all_infos.extend(info)
        return all_infos

    def create_groundtruth_database(
        self, info_path=None, used_classes=None, split="train"
    ):
        import torch

        database_save_path = Path(self.root_path) / (
            "gt_database" if split == "train" else ("gt_database_%s" % split)
        )
        db_info_save_path = Path(self.root_path) / ("once_dbinfos_%s.pkl" % split)

        all_db_infos = {}

        infos = self.client.load_pickle(info_path)

        for k in range(len(infos)):
            if "annos" not in infos[k]:
                continue
            print("gt_database sample: %d" % (k + 1))
            info = infos[k]
            frame_id = info["frame_id"]
            seq_id = info["sequence_id"]
            points = self.get_lidar(seq_id, frame_id)
            annos = info["annos"]
            names = annos["name"]
            gt_boxes = annos["boxes_3d"]

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = "%s_%s_%d.bin" % (frame_id, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                self.client.put(gt_points.tobytes(), filepath)

                db_path = str(
                    filepath.relative_to(self.root_path)
                )  # gt_database/xxxxx.bin
                db_info = {
                    "name": names[i],
                    "path": db_path,
                    "gt_idx": i,
                    "box3d_lidar": gt_boxes[i],
                    "num_points_in_gt": gt_points.shape[0],
                }
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

        for k, v in all_db_infos.items():
            print("Database %s: %d" % (k, len(v)))

        self.client.dump_pickle(all_db_infos, db_info_save_path)

    @staticmethod
    def generate_prediction_dicts(
        batch_dict, pred_dicts, class_names, output_path=None
    ):
        def get_template_prediction(num_samples):
            ret_dict = {
                "name": np.zeros(num_samples),
                "score": np.zeros(num_samples),
                "boxes_3d": np.zeros((num_samples, 7)),
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
            pred_dict["boxes_3d"] = pred_boxes
            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict["frame_id"][index]
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict["frame_id"] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                raise NotImplementedError
        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        from .once_eval.evaluation import get_evaluation_results

        eval_det_annos = copy.deepcopy(det_annos)
        # eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.once_infos]
        eval_gt_annos = [
            copy.deepcopy(self.once_infos[itv[1] - 1]["annos"])
            for itv in self.once_intervals
        ]
        ap_result_str, ap_dict = get_evaluation_results(
            eval_gt_annos, eval_det_annos, class_names
        )

        return ap_result_str, ap_dict


def create_once_infos(dataset_cfg, class_names, data_path, save_path, workers=32):
    raise NotImplementedError


def vis_samples(
    dataset_cfg, class_names, data_path, processed_data_tag="waymo_processed_data"
):
    from torch.utils.data import DataLoader

    dataset = ONCETemporalDataset(
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=data_path,
        training=True,
        logger=common_utils.create_logger(),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        pin_memory=True,
        num_workers=0,
        shuffle=False,
        collate_fn=dataset.collate_batch,
        drop_last=True,
        sampler=None,
        timeout=0,
    )

    for batch_dict in dataloader:
        print(batch_dict.keys())
        print(batch_dict["frame_id"])
        print(batch_dict["points"].shape)
        print(batch_dict["gt_boxes"].shape)

        # batch_dict['points']: [N_points, 5], where 5 = [batch_idx, x, y, z, intensity]
        # if bs>1, special points are inserted to separate frames
        points = batch_dict["points"][:, 1:]
        # batch_dict['gt_boxes']: [batch_idx, N_gt, 8], 7 = [x, y, z, w, l, h, ry, box_type]
        # columns 7 is box type, e.g. Car, Pedestrian, Cyclist
        gt_boxes = batch_dict["gt_boxes"][0][:, :7]  # batch size set to 1

        # draw_scenes(points, gt_boxes=gt_boxes)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file", type=str, default=None, help="specify the config of dataset"
    )
    parser.add_argument("--func", type=str, default="create_waymo_infos", help="")
    args = parser.parse_args()

    if args.func == "vis_samples":
        cfg_file = args.cfg_file

        import yaml
        from pathlib import Path
        from easydict import EasyDict

        dataset_cfg = EasyDict(yaml.safe_load(open(cfg_file)))

        ROOT_DIR = (Path(__file__).resolve().parent / "../../../").resolve()
        once_data_path = ROOT_DIR / "data" / "once"
        once_save_path = ROOT_DIR / "data" / "once"

        vis_samples(
            dataset_cfg=dataset_cfg,
            class_names=["Car", "Bus", "Truck", "Pedestrian", "Cyclist"],
            data_path=once_data_path,
        )
