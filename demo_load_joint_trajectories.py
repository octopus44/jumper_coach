from data_utils import load_pickle

pkl_path = "data/joint_trajectories.pkl"
joint_dict = load_pickle(pkl_path)

# get joint trajectory for a specific video
video_name = "C0001"
print(joint_dict[video_name].shape) # np.array([TxKx3])
# print(joint_dict[video_name])