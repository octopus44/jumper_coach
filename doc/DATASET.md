## Prepare Dataset

The following command extracts joint trajectories from the 3D pose predictor output directories. 
The joint trajectories are saved to a dictionary with the form `{"video_name": np.array([T,K,3])}` 
where `T` is the trajectory length and `K` is the number of joints.

The dictionary is stored in a pickle file called `joint_trajectories.pkl`.
```
python3 prepare_dataset.py --in_dir /path/to/pose_predictions/ --out_dir data/
```

## Loading Joint Trajectories
The joint trajectories can be loaded into python by simply unpacking the pickle file.
```
from data_utils import load_pickle

pkl_path = "data/joint_trajectories.pkl"
joint_dict = load_pickle(pkl_path)

# get joint trajectory for a specific video
video_name = "C0001"
print(joint_dict[video_name].shape) # np.array([TxKx3])
print(joint_dict[video_name]) 
``` 