import os
import numpy as np
import torch

from common.quaternion import qbetween_np, qrot_np
from common.skeleton import Skeleton
from paramUtil import *

t2m_raw_offsets = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 0, 1],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
    ]
)


class MotionPreprocess:
    def __init__(self, data_dir="/liujinxin/code/HumanML3D/joints"):
        self.kinematic_chain = t2m_kinematic_chain
        self.n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
        self.l_idx1 = 5
        self.l_idx2 = 8
        self.face_joint_indx = [2, 1, 17, 16]

        example_id = "000021"
        example_data = np.load(os.path.join(data_dir, example_id + ".npy"))
        example_data = example_data.reshape(len(example_data), -1, 3)
        example_data = torch.from_numpy(example_data)
        tgt_skel = Skeleton(self.n_raw_offsets, self.kinematic_chain, "cpu")
        # (joints_num, 3)
        self.tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])

    def uniform_skeleton(self, positions, target_offset):
        src_skel = Skeleton(self.n_raw_offsets, self.kinematic_chain, "cpu")
        src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
        src_offset = src_offset.numpy()
        tgt_offset = target_offset.numpy()
        # print(src_offset)
        # print(tgt_offset)
        """Calculate Scale Ratio as the ratio of legs"""
        l_idx1 = self.l_idx1
        l_idx2 = self.l_idx2
        src_leg_len = (
            np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
        )
        tgt_leg_len = (
            np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()
        )

        scale_rt = tgt_leg_len / src_leg_len
        # print(scale_rt)
        src_root_pos = positions[:, 0]
        tgt_root_pos = src_root_pos * scale_rt

        """Inverse Kinematics"""
        quat_params = src_skel.inverse_kinematics_np(positions, self.face_joint_indx)
        # print(quat_params.shape)

        """Forward Kinematics"""
        src_skel.set_offset(target_offset)
        new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
        return new_joints

    def process_file(self, positions, feet_thre=0.002):
        # support 22 joints
        assert positions.shape[1] == 22

        if check_left_right(positions):
            print("!!! left and right are swapped, swap them")
            positions = swap(positions)
            check_left_right(positions)

        # (seq_len, joints_num, 3)
        #     '''Down Sample'''
        #     positions = positions[::ds_num]

        """Uniform Skeleton"""
        positions = self.uniform_skeleton(positions, self.tgt_offsets)

        """Put on Floor"""
        floor_height = positions.min(axis=0).min(axis=0)[1]
        positions[:, :, 1] -= floor_height
        #     print(floor_height)

        #     plot_3d_motion("./positions_1.mp4", kinematic_chain, positions, 'title', fps=20)

        """XZ at origin"""
        root_pos_init = positions[0]
        root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
        positions = positions - root_pose_init_xz

        # '''Move the first pose to origin '''
        # root_pos_init = positions[0]
        # positions = positions - root_pos_init[0]

        """All initially face Z+"""
        r_hip, l_hip, sdr_r, sdr_l = self.face_joint_indx
        across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
        across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
        across = across1 + across2
        across = across / np.sqrt((across**2).sum(axis=-1))[..., np.newaxis]

        # forward (3,), rotate around y-axis
        forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        # forward (3,)
        forward_init = (
            forward_init / np.sqrt((forward_init**2).sum(axis=-1))[..., np.newaxis]
        )

        #     print(forward_init)

        target = np.array([[0, 0, 1]])
        root_quat_init = qbetween_np(forward_init, target)
        root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

        positions = qrot_np(root_quat_init, positions)

        return positions


def check_left_right(motion):
    root_pos_init = motion[0]
    r_hip, l_hip, sdr_r, sdr_l = [2, 1, 17, 16]
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across**2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)

    toe_across1 = root_pos_init[10] - root_pos_init[7]
    toe_across2 = root_pos_init[11] - root_pos_init[8]
    toe_across = toe_across1 + toe_across2
    toe_across = toe_across / np.sqrt((toe_across**2).sum(axis=-1))[..., np.newaxis]

    # check 两个向量是不是在一个方向上
    print(np.dot(forward_init[0], toe_across))
    if np.dot(forward_init[0], toe_across) < 0:
        return True
    return False


def swap(motion):
    m = motion.copy()
    m[:, 13] = motion[:, 14]
    m[:, 14] = motion[:, 13]

    m[:, 16] = motion[:, 17]
    m[:, 17] = motion[:, 16]

    m[:, 18] = motion[:, 19]
    m[:, 19] = motion[:, 18]

    m[:, 20] = motion[:, 21]
    m[:, 21] = motion[:, 20]

    m[:, 1] = motion[:, 2]
    m[:, 2] = motion[:, 1]

    m[:, 4] = motion[:, 5]
    m[:, 5] = motion[:, 4]

    m[:, 7] = motion[:, 8]
    m[:, 8] = motion[:, 7]

    m[:, 10] = motion[:, 11]
    m[:, 11] = motion[:, 10]
    return m


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "/liujinxin/code/HumanML3D")
    from dataset_utils import MotionPreprocess

    processor = MotionPreprocess()

    file = "/liujinxin/dataset/BABEL/motion/KIT_348_walking_run07_poses.npz.npy"
    # file = "/liujinxin/dataset/BABEL/motion/EKUT_234_MTR03_poses.npz.npy"
    # file = "/ssdwork/liujinxin/DATASET/Hu/HumanML3D/new_joints/012323.npy"
    motion = np.load(file)
    motion = motion[:, :22]

    result = processor.process_file(motion)
    np.save(
        "/liujinxin/code/Hu_mjf/output/KIT_348_walking_run07_poses.npz.npy_p.npy",
        result,
    )
    print(result.shape)
