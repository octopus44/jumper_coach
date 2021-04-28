#!/usr/bin/env python
import os
import numpy as np
import pickle


def reflect_joints3d(joints):
    # joints is TxKx3
    # T: trajectory length
    # K: number of joints

    flip_mat = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    joints_flipped = np.dot(flip_mat,joints[:,:,:,np.newaxis]).T[0]
    joints_flipped=np.moveaxis(joints_flipped, [1, 0], [0,1])

    return joints_flipped


def get_dir_list(path):
    return [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


def save_pickle(data,pkl_path):
    pickle.dump(data,open(pkl_path,'wb'))


def load_pickle(pkl_path):
    data = pickle.load(open(pkl_path,'rb'))
    return data