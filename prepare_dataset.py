#!/usr/bin/env python
import os
import argparse
import copy
import data_utils
import numpy as np


def load_pose_predictions(input_dir):
    pred_dirs = data_utils.get_dir_list(input_dir)
    pred_dict = {}
    for d in pred_dirs:
        vid_name = d.split("/")[-1]
        pred_dict[f"{vid_name}"]= data_utils.load_pickle(os.path.join(d,"hmmr_output/hmmr_output.pkl"))
    return pred_dict


def load_annotations(input_file):
    ann_dict = data_utils.load_dict_from_csv(input_file)
    # convert list of dicts to dict of lists (indexed by name)

    return ann_dict

def trim_trajectories(pred_dict,ann_dict):
    pred_dict_trim = copy.copy(pred_dict)
    for i, vid_name in enumerate(ann_dict["name"]):
        if vid_name in pred_dict_trim.keys():
            pred = pred_dict_trim[vid_name]
            # trim trajectory to specified range
            t0 = int(ann_dict["start_frame"][i])
            tf = int(ann_dict["end_frame"][i])

            if tf > pred["joints"].shape[0] or tf < 0:
                print(
                    f"WARNING: Skipping {vid_name}. \'end_frame\' {tf} is out of range for trajectory of length {pred['joints'].shape[0]}.")
                continue
            if t0 > pred["joints"].shape[0] or t0 < 0:
                print(
                    f"WARNING: Skipping {vid_name}. \'start_frame\' {t0} is out of range for trajectory of length {pred['joints'].shape[0]}.")
                continue

            for key,traj in pred.items():
                pred_dict_trim[vid_name][key] = traj[t0:tf,...]
    return pred_dict_trim


def get_runup_angle(joints,n=10):
    # poses is (Tx(K+1)x3x3), first pose is global
    angles = np.arctan2((joints[0:n,2,2] - joints[0:n,3,2]),(joints[0:n,2,0] - joints[0:n,3,0]))# assume first second or so is the run up
    return np.mean(angles)

def normalize_viewpoint(joints, des_theta = -np.pi/2):
    cur_theta = get_runup_angle(joints,n=10)
    dtheta = cur_theta - des_theta

    joints_rot = data_utils.rotate_trajectory_about_y(joints,dtheta)
    return joints_rot

def save_pose_trajectories(pred_dict,output_dir):
    pose_dict = {}
    for vid_name,pred in pred_dict.items():
        pose_dict[vid_name] = pred["poses"]
    data_utils.save_pickle(pose_dict,os.path.join(output_dir,"pose_trajectories.pkl"))


def save_cam_trajectories(pred_dict,output_dir):
    cam_dict = {}
    for vid_name,pred in pred_dict.items():
        cam_dict[vid_name] = pred["cams"]
    data_utils.save_pickle(cam_dict,os.path.join(output_dir,"cam_trajectories.pkl"))


def save_joint_trajectories(pred_dict,output_dir):
    # TODO: Figure out which joints we actually want
    """
        joints is 27x3. but if not will transpose it.
        0: Right heel
        1: Right knee
        2: Right hip
        3: Left hip
        4: Left knee
        5: Left heel
        6: Right wrist
        7: Right elbow
        8: Right shoulder
        9: Left shoulder
        10: Left elbow
        11: Left wrist
        12: Neck
        13: Head top
        14: nose
        15: left_eye
        16: right_eye
        17: left_ear
        18: right_ear
        19: left big toe
        20: right big toe
        21: Left small toe
        22: Right small toe
        23: L ankle
        24: R ankle
    """
    joints_dict = {}
    joints_dict_norm = {}
    for vid_name,pred in pred_dict.items():
        joints_dict[vid_name] = pred["joints"]
        joints_dict_norm[vid_name] = normalize_viewpoint(pred["joints"])
    data_utils.save_pickle(joints_dict,os.path.join(output_dir,"joint_trajectories.pkl"))
    data_utils.save_pickle(joints_dict_norm,os.path.join(output_dir,"joint_trajectories_norm.pkl"))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_dir', action='store', dest='input_dir',
                        help='Input pose predictions directory. ')
    parser.add_argument('--out_dir', action='store', dest='out_dir',
                        help='Directory to store joint trajectories')
    args = parser.parse_args()

    input_dir = args.input_dir
    out_dir = args.out_dir

    pose_preds = load_pose_predictions(input_dir)
    ann_dicts = load_annotations("data/annotations.csv")

    pose_preds_trimmed = trim_trajectories(pose_preds,ann_dicts)

    save_joint_trajectories(pose_preds_trimmed,out_dir)
    save_pose_trajectories(pose_preds_trimmed,out_dir)
    save_cam_trajectories(pose_preds_trimmed,out_dir)


if __name__ == '__main__':
    main()