import sys, os
import torch
import numpy as np


from human_body_prior.tools.omni_tools import copy2cpu as c2c

os.environ["PYOPENGL_PLATFORM"] = "egl"


# Choose the device to run the body model on.
comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from human_body_prior.body_model.body_model import BodyModel

male_bm_path = "./body_model/smplh/male/model.npz"
male_dmpl_path = "./body_model/dmpls/male/model.npz"

female_bm_path = "./body_model/smplh/female/model.npz"
female_dmpl_path = "./body_model/dmpls/female/model.npz"

num_betas = 10  # number of body parameters
num_dmpls = 8  # number of DMPL parameters

male_bm = BodyModel(
    bm_fname=male_bm_path,
    num_betas=num_betas,
    num_dmpls=num_dmpls,
    dmpl_fname=male_dmpl_path,
).to(comp_device)
faces = c2c(male_bm.f)

female_bm = BodyModel(
    bm_fname=female_bm_path,
    num_betas=num_betas,
    num_dmpls=num_dmpls,
    dmpl_fname=female_dmpl_path,
).to(comp_device)


trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
ex_fps = 20


def amass_to_pose(src_path, save_path):
    bdata = np.load(src_path, allow_pickle=True)
    fps = 0
    try:
        fps = bdata["mocap_framerate"]
        frame_number = bdata["trans"].shape[0]
    except:
        #         print(list(bdata.keys()))
        return fps

    fId = 0  # frame id of the mocap sequence
    pose_seq = []
    if bdata["gender"] == "male":
        bm = male_bm
    else:
        bm = female_bm
    down_sample = int(fps / ex_fps)
    #     print(frame_number)
    #     print(fps)

    bdata_poses = bdata["poses"][::down_sample, ...]
    bdata_trans = bdata["trans"][::down_sample, ...]
    body_parms = {
        "root_orient": torch.Tensor(bdata_poses[:, :3]).to(comp_device),
        "pose_body": torch.Tensor(bdata_poses[:, 3:66]).to(comp_device),
        "pose_hand": torch.Tensor(bdata_poses[:, 66:]).to(comp_device),
        "trans": torch.Tensor(bdata_trans).to(comp_device),
        "betas": torch.Tensor(
            np.repeat(
                bdata["betas"][:num_betas][np.newaxis], repeats=len(bdata_trans), axis=0
            )
        ).to(comp_device),
    }

    with torch.no_grad():
        body = bm(**body_parms)
    pose_seq_np = body.Jtr.detach().cpu().numpy()
    pose_seq_np_n = np.dot(pose_seq_np, trans_matrix)

    np.save(save_path, pose_seq_np_n)
    return fps
