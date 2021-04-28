#!/usr/bin/env python
import os
import argparse

from data_utils import reflect_joints3d,get_dir_list,save_pickle,load_pickle



def load_pose_predictions(input_dir):
    pred_dirs = get_dir_list(input_dir)
    pred_dict = {}
    for d in pred_dirs:
        vid_name = d.split("/")[-1]
        pred_dict[f"{vid_name}"]= load_pickle(os.path.join(d,"hmmr_output/hmmr_output.pkl"))
    return pred_dict

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
        (added)
        25: hip midpoint
        26: shoulder midpoint
    """
    joints_dict = {}
    for vid_name,pred in pred_dict.items():
        joints_reflected = reflect_joints3d(pred["joints"])
        joints_dict[vid_name] = joints_reflected

    save_pickle(joints_dict,os.path.join(output_dir,"joint_trajectories.pkl"))


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

    save_joint_trajectories(pose_preds,out_dir)



if __name__ == '__main__':
    main()