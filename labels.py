import os
import unittest
from sys import argv
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
from data_utils import plot_skeleton_3d


def get_data(filename):
    jump_video = pd.read_csv((f'data/{filename}'), sep=',', encoding="UTF8", quotechar='"', skipinitialspace=True, header=0)

    with open('data/joint_trajectories.pkl', 'rb') as f:
        joint_dict = pkl.load(f)

    with open('data/joint_trajectories_norm.pkl', 'rb') as f:
        joint_t_n = pkl.load(f)

    with open('data/pose_trajectories.pkl', 'rb') as f:
        pose = pkl.load(f)
    return jump_video, joint_dict, joint_t_n, pose


def show_position(video_name,joint_dict):
    # video_name = "C0001"
    print(joint_dict[video_name].shape) # np.array([TxKx3])
    print(joint_dict[video_name])
    print(len(joint_dict))

def store_jumper():
    count = 0
    with open('insert_ref_data.sql', 'r') as fm:
        Lines = fm.readlines()
        for line in Lines:
            count += 1
            print("Line{}: {}".format(count, line.strip()))


def store_videos(joint_dict,workfile):
    with open(workfile,'w') as f:
        s = (f'SET TIMEZONE = "America/New_York";\n')
        f.write(s)
        store_jumpers(f)
        for i in joint_dict.keys():
            s = (f"insert into jump_part (jump_part_code ) values " +
                 f"('{i}');\n")
            f.write(s)


def main(arg):
    script, filename, debug = argv
    print("================================")
    if debug == 'False': debug = bool(0)
    if debug == 'True': debug = bool(1)
    if debug: print({debug})
    print(f'Jumper Project Home Dir:')
    os.system("pwd")
    print("================================")
    # get joint trajectory for a specific video
    jump_video, joint_dict, joint_t_n, pose = get_data(filename)
    plot_skeleton_3d(joint_dict["C0032b"][96 - 17, :, :])
    store_videos(joint_dict,'save_me.csv')
    # plt.show()



