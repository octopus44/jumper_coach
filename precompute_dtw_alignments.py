from data_utils import get_dir_list,save_pickle,load_pickle,plot_skeleton_3d,rot_mat_to_euler, load_dict_from_csv
from dtw import *
import matplotlib.pyplot as plt
import numpy as np
import os

data_path = "data"
joint_dict = load_pickle(os.path.join(data_path,"joint_trajectories.pkl"))
joint_dict_norm = load_pickle(os.path.join(data_path,"joint_trajectories_norm.pkl"))
pose_dict = load_pickle(os.path.join(data_path,"pose_trajectories.pkl"))
ann_dict = load_dict_from_csv(os.path.join(data_path,"annotations.csv"))


dtw_alignment_dict = {}
for name1 in joint_dict_norm.keys():
    this_alignment_dict = {}
    for name2 in joint_dict_norm.keys():
        if name1 != name2:

            X1 = joint_dict_norm[name1].reshape((joint_dict_norm[name1].shape[0],joint_dict_norm[name1].shape[1]*joint_dict_norm[name1].shape[2]))
            X2 = joint_dict_norm[name2].reshape((joint_dict_norm[name2].shape[0],joint_dict_norm[name2].shape[1]*joint_dict_norm[name2].shape[2]))
            alignment = dtw(X1, X2, keep_internals=True)
            this_alignment_dict[name2] = {"index1":alignment.index1,
                                          "index2":alignment.index2,
                                          "dist":alignment.distance}

    dtw_alignment_dict[name1] = this_alignment_dict

save_pickle(dtw_alignment_dict,os.path.join("data","dtw_alignments.pkl"))


# plot_skeleton_3d(joint_dict_norm["C0023b"][0,:,:])
# plot_skeleton_3d(joint_dict["C0023b"][0,:,:])
# plot_skeleton_3d(joint_dict_norm["C0010b"][0,:,:])
# plot_skeleton_3d(joint_dict["C0010b"][12,:,:])
plot_skeleton_3d(joint_dict["C0002"][72,:,:])
print(joint_dict["C0001"][53,:,:])
plt.show()



