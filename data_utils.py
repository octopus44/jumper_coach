#!/usr/bin/env python
import os
import numpy as np
import pickle
import csv
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


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


def get_dir_list(path):
    return [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


def save_pickle(data,pkl_path):
    pickle.dump(data,open(pkl_path,'wb'))


def load_pickle(pkl_path):
    data = pickle.load(open(pkl_path,'rb'))
    return data


def reflect_pose(poses):
    # joints is TxKx3x3
    # T: trajectory length
    # K: number of joints
    flip_mat = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    poses_flipped = np.matmul(flip_mat,poses[...,np.newaxis])[...,0]

    return poses_flipped

def reflect_joints3d(joints):
    # joints is TxKx3
    # T: trajectory length
    # K: number of joints
    flip_mat = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    joints_flipped = np.dot(flip_mat,joints[...,np.newaxis]).T[0]
    joints_flipped=np.moveaxis(joints_flipped, [1, 0], [0,1])


    return joints_flipped


def plot_skeleton_3d(joints,ax=None, limb_color= '#1f77b4',joint_color='#d62728',marker_size=15,linewidth=3,
                     xrange=[-1,1],yrange=[-1,1],zrange=[-1,1]):
    """
        joints is 27x3. (i.e. single timestep)
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
    # add hip and shoulder midpoints
    joints = np.concatenate(
        (joints, (joints[ 2, :] + joints[ 3, :])[ np.newaxis, :] / 2), axis=0)
    joints = np.concatenate(
        (joints, (joints[ 8, :] + joints[ 9, :])[ np.newaxis, :] / 2), axis=0)

    if ax is None:
        fig = plt.figure()
        ax = plt.axes(projection='3d')

    # Joints that should be connected (in order)
    conns = [[22, 20, 0, 24, 1, 2, 25, 3, 4, 23, 5, 19, 21], [6, 7, 8, 26, 9, 10, 11], [25, 26, 12, 13], [17, 13, 18]]
    skip_joint = [14, 15, 16]
    skip_mask = [not (j in skip_joint) for j in range(joints.shape[0])]

    # draw joint lines

    for conn in conns:
        ax.plot3D(joints[conn, 0], joints[conn, 2],-1 * joints[conn, 1],c=limb_color,linestyle='-',linewidth=linewidth)
    ax.scatter3D(joints[skip_mask,0],joints[skip_mask,2],-1*joints[skip_mask,1],marker="*",c=joint_color,s=marker_size)
    # set equal aspect ratio

    ax.set_xlim(xrange[0], xrange[1])
    ax.set_zlim(yrange[0], yrange[1])
    ax.set_ylim(zrange[0], zrange[1])
    #note: z and y switched on purpose
    ax.set_box_aspect(aspect=(xrange[1] - xrange[0], zrange[1] - zrange[0], yrange[1] - yrange[0]),zoom=1)
    ax.set_xlabel('x')
    ax.set_zlabel('y')
    ax.set_ylabel('z')
    return ax


def rotate_trajectory_about_y(joints,theta):
    # joints is (TxKx3)
    # theta is the desired rotation around y to apply to joints
    Ry = np.array([[np.cos(theta) , 0, np.sin(theta)],
                   [0             , 1, 0            ],
                   [-np.sin(theta), 0, np.cos(theta)]])
    joints_rot = np.dot(Ry, joints[:, :, :, np.newaxis]).T[0]
    joints_rot = np.moveaxis(joints_rot, [1, 0], [0, 1])
    return joints_rot


def rot_mat_to_euler(R):
    theta_x = np.arctan2(R[2,1],R[2,2])
    theta_y = np.arctan2(-R[2,0],np.sqrt(R[2,1]**2 + R[2,2]**2))
    theta_z = np.arctan2(R[1,0],R[0,0])
    return theta_x,theta_y,theta_z