import os
import csv
import random
import pickle
from sklearn.model_selection import KFold

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from typing import TypeVar
T_co = TypeVar('T_co', covariant=True)


def match_data(trajectories='data/joint_trajectories_norm.pkl', annotations='data/annotations.csv', output_file='data/dataset_norm.pkl'):
    """
    To match joint trajectory predictions & annotations (success or fail) by video sequence name.
    
    Inputs:
    - trajectories(str) : directory of joint trajectory pickle file.
    - annotations (str) : directory of annotations csv file.
    - output_file (str) : desired output directory and filename (create matched pickle file)
    
    Output: *Saves new pickle file at specified directory*
    Format: {'C0001': {'pose': np.ndarray(T, 25, 3), 'label' : int(1 or 0)}, 
             'C0002': {'pose': np.ndarray(T, 25, 3), 'label' : int(1 or 0)},
             ......
    }
    """
    
    pred_dict = pickle.load(open(os.path.join(trajectories), "rb"))
    #for key in pred_dict.keys():
    #    print(key, pred_dict[key].shape) #T, 25, 3

    with open(annotations, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        anno = []
        for row in spamreader:
            anno.append(row)
        
    anno = anno[1:] # remove header
    
    # match sequences with annos
    dataset = {}
    for key in pred_dict.keys():
        for a in anno:
            if key == a[0]:
                dataset[key] = {'pose': pred_dict[key], 'label': int(a[3]), 'direction' : 0 if a[4]=='right' else 1}
            
    with open(output_file, 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

class Subset(Dataset[T_co]):
    """
    For splitting datasets (at each fold of cross validation).
    Directly adapted from `torch.utils.data.dataset` source code.
    """
    
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

def kfold_split(dataset, n_splits=10): 
    """
    Modified from source code of `torch.utils.data.random_split`.
    for K-Fold cross validation.
    
    Example:
    kfold_dataset = kfold_split(dataset, n_splits=10)
    for k, v in kfold_dataset:
        trainloader = DataLoader(k, batch_size=4)
        validloader = DataLoader(v, batch_size=4)
        
    KFold: `sklearn.model_selection.KFold` object. Returns [train, valid] lists of indices for each fold / split.
    
    Inputs:
    - dataset (Dataset) : Created with the following functions.
    - n_splits(int)     : Number of partitions for cross validation.
    
    Outputs:
    - rounds(List[List[Subset, Subset]]) : Train and val subsets for each fold. A list of length n_fold.
    """
    
    kfold = KFold(n_splits=n_splits)
    rounds = []
    for fold, (train_index, val_index) in enumerate(kfold.split(dataset)):
        rounds.append([Subset(dataset, train_index), Subset(dataset, val_index)])
        
    return rounds
    
class Frames(Dataset):
    """
    Main dataset.
    
    Inputs:
    dataset_pkl_file(str) : Path to preprocessed pickle file.
    
    Outputs (__getitem__ returns):
    joints     : np.ndarray(T, 75)
    labels     : int(1 or 0)
    directions : int(0 for left and 1 for right)
    names      : str(sequence name, for reference only, e.g. 'C0001')}
    """
    
    def __init__(self, dataset_pkl_file='data/dataset_norm.pkl'):
        with open(dataset_pkl_file, 'rb') as handle:
            data = pickle.load(handle)
            
        self.joints = []# all seq files, num_videos * T(var) * 75
        self.labels = []# labels, 0 or 1
        self.names = []
        self.directions = []
        for key in data.keys():
            self.joints.append(data[key]['pose'].reshape(-1, 75))  # T * 75
            self.labels.append(data[key]['label'])
            self.names.append(key)
            self.directions.append(data[key]['direction']) # int
            
        temp = list(zip(self.joints, self.labels, self.names, self.directions))
        random.shuffle(temp)
        self.joints, self.labels, self.names, directions = zip(*temp)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.joints[idx], self.labels[idx], self.directions[idx], self.names[idx] # use collate to deal with different lengths
    
def collate_fn(batch):
    """
    Collate function.
    Processes the data after Frame.__getitem__, in order to pass them into dataloaders.
    
    Inputs: Batched data, [x, y, n] * batch_size.
    - joints (np.ndarray) : batch_size * T(var) * (25 * 3)
    - labels (int)        : batch_size * 1
    - directions (int)    : batch_size * 1
    - names (str)         : filename of sequence ('C0001' etc.), batch_size * 1
    
    Outputs: Stacks padded data into batches.
    - ret(dict): {
        - x_pad    (torch.Tensor)     : batch_size * max_T * 75 
        - x_lens   (torch.LongTensor) : batch_size * 1 (each T of x)
        - y        (torch.LongTensor) : batch_size * 1
        - dir      (torch.LongTensor) : batch_size * 1
        - filename (str)              : batch_size * 1
      }
    
    We have to pad the sequences with different lengths here & record their lengths respectively, for the dataloader & RNNs to work.
    """
    
    joints  = [item[0] for item in batch]
    labels = torch.LongTensor([item[1] for item in batch])
    directions = torch.LongTensor([item[2] for item in batch])
    names  = [item[3] for item in batch]

    x = [torch.Tensor(xx) for xx in joints]

    lens = torch.LongTensor([len(xx) for xx in joints])
    x_pad = pad_sequence([xx for xx in x], batch_first=True, padding_value=0)

    ret = {'x' : x_pad, 'x_lens' : lens, 'y': labels, 'dir': directions, 'filename' : names}
    return ret
