import os
import unittest
from sys import argv
import pickle as pkl


with open('data/cam_trajectories.pkl', 'rb') as f:
    joint_dict = pkl.load(f)

with open('data/dtw_alignments.pkl', 'rb') as f:
    dtw_a = pkl.load(f)

with open('data/joint_trajectories.pkl', 'rb') as f:
    joint_t = pkl.load(f)

with open('data/joint_trajectories_norm.pkl', 'rb') as f:
    joint_t_n = pkl.load(f)

with open('data/pose_trajectories.pkl', 'rb') as f:
    pose = pkl.load(f)

# get joint trajectory for a specific video
video_name = "C0001"
print(joint_dict[video_name].shape) # np.array([TxKx3])
print(joint_dict[video_name])
print(len(joint_dict))

def main(arg):
    script, filename, outfile, debug = argv
    print("================================")
    script, debug = argv
    if debug == 'False': debug = bool(0)
    if debug == 'True': debug = bool(1)
    if debug: print({debug})
    os.system("pwd")
    print("================================")

if __name__ == "__main__":
    main(argv)
    unittest.main(argv)