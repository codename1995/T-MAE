import numpy as np
from scipy.spatial.transform import Rotation

def convert_prv_frame_to_cur(pc_prv, pose_prv, pose_cur):
    # In the poses format that ONCE uses, if the ego vehicle is static, the pose is [0, 0, 0, 0, 0, 0, 0]
    # In this case, we should not do any transformation to the point cloud

    # Convert pc_prv to global
    if np.any(pose_prv) == 0:  # if all zeros, skip
        pc_prv_global = pc_prv[:, :3]
    else:
        R = Rotation.from_quat(pose_prv[:4]).as_matrix()
        t = np.array(pose_prv[4:]).transpose()
        pc_prv_global = np.dot(pc_prv[:, :3], R.T) + t

    # Convert pc_prv_global to current coordinate system
    if np.any(pose_cur) == 0:  # if all zeros, skip
        pass
    else:
        R_global2cur_44 = np.zeros((4, 4))
        R_global2cur_44[:3, :3] = Rotation.from_quat(pose_cur[:4]).as_matrix()
        R_global2cur_44[:3, 3] = np.array(pose_cur[4:]).transpose()
        R_global2cur_44[3][3] = 1
        R_global2cur_44 = np.linalg.inv(R_global2cur_44)
        expand_pc_prv_global = np.concatenate([pc_prv_global, np.ones((pc_prv_global.shape[0], 1))], axis=-1)
        pc_prv_global = np.dot(expand_pc_prv_global, R_global2cur_44.T)[:, :3]

    pc_prv2cur = np.concatenate([pc_prv_global[:, :3], pc_prv[:, 3:]], axis=-1)
    return pc_prv2cur

def convert_to_global(pc, pose):
    expand_pc = np.concatenate([pc[:, :3], np.ones((pc.shape[0], 1))], axis=-1)
    pc_global = np.dot(expand_pc, pose.T)[:, :3]
    pc_global = np.concatenate([pc_global, pc[:, 3:]], axis=-1)
    return pc_global

def convert_to_local(global_pc, pose):
    expand_pc_global = np.concatenate([global_pc[:, :3], np.ones((global_pc.shape[0], 1))], axis=-1)
    pc_local = np.dot(expand_pc_global, np.linalg.inv(pose.T))[:, :3]
    pc_local = np.concatenate([pc_local, global_pc[:, 3:]], axis=-1)
    return pc_local

def remove_ego_points(points, center_radius=1.0):
    mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
    return points[mask]
