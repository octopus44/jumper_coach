import csv
import os
import unittest
from sys import argv
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import data_utils


def load_dict_from_csv(input_file):
    # assumes header in first line
    out = None
    with open(input_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if out is None:
                out = {}
                for key,val in row.items():
                    out[key] = [val]
            else:
                for key,val in row.items():
                    out[key].append(val)

    return out


def load_pickle(pkl_path):
    data = pkl.load(open(pkl_path,'rb'))
    return data

def distance(a, b):
    """
    Returns the Euclidean distance between a and b
    """
    aa = np.array(a)
    bb = np.array(b)
    distant = np.linalg.norm(aa-bb)
    return distant


def get_video_knees(dict, name, debug):
    try:
        test_video = dict[name]
    except KeyError:
        if debug: print(f' video {name} not found ')
        test_video = 0

    rt_knee = dict[test_video][53,1,:]
    lt_knee = dict[test_video][53,4,:]
    return test_video, rt_knee, lt_knee


def low_level_recognize(joint_dict, name, debug):
    rt_knee, lt_knee = get_video_knees(joint_dict, name, debug)
    knee_distance = distance(rt_knee, lt_knee)
    data_utils.plot_skeleton_3d(joint_dict["C0002"][53, :, :])
    return knee_distance


def high_level_recognize():
    pass

def main(arg):
    script,  debug = argv
    # filename, outfile,
    print("================================")
    script, debug = argv
    if debug == 'False': debug = bool(0)
    if debug == 'True': debug = bool(1)
    if debug: print({debug})
    os.system("pwd")
    print("================================")
    data_path = "data"
    joint_dict = load_pickle(os.path.join(data_path, "joint_trajectories.pkl"))
    joint_dict_norm = load_pickle(os.path.join(data_path, "joint_trajectories_norm.pkl"))
    pose_dict = load_pickle(os.path.join(data_path, "pose_trajectories.pkl"))
    ann_dict = load_dict_from_csv(os.path.join(data_path, "annotations.csv"))
    s = []
    for i in range(len(ann_dict['name'])):
        video_name = ann_dict['name'][i]
        knee_distance = low_level_recognize(joint_dict, video_name, debug)

if __name__ == "__main__":
    main(argv)
    unittest.main(argv)