import os
import csv
import random
import pickle
import numpy as np
from sklearn.model_selection import KFold
from collections import defaultdict, Counter

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from typing import TypeVar
T_co = TypeVar('T_co', covariant=True)

def parse_labels(labels_str):
    """
    Parse strings from .csv file to list of string labels.
    Inputs:
    labels_str   (str) : A big string of labels, loaded from .csv file, e.g. "['a'; 'b'; 'c']"
    
    Outputs:
    labels (list[str]) : List of original string labels, e.g. ['a', 'b', 'c']. 
    """
    
    labels = [l.replace("\'","").strip(" ") for l in labels_str.strip("[\']").split(";")]
    return labels

def match_labels(trajectories='data/joint_trajectories_norm.pkl',
                 annotations='data/annotations.csv', 
                 output_file='data/dataset_with_labels.pkl'):
    
    """
    Match predicted trajectories and annotations.
    
    Input:
    trajectories (str) : .pkl file of trajectory predictions. 
    annotations  (str) : .csv file of annotation.
    output_file  (str) : .pkl file of destination. 
    
    Output: 
    None (write to new .pkl file).
    """

    # Load prediction file
    pred_dict = pickle.load(open(os.path.join(trajectories), "rb")) # 'filename': T * 25 * 3
    
    # Load label file
    with open('data/labels.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        anno = []
        for row in spamreader:
            anno.append(row)
        
    anno = anno[1:] # remove header

    dataset = {}
    i = 0
    for key in pred_dict.keys(): # all video sequences
        for row in anno:
            if key == row[0]:
                sid = int(row[1])
                rid = [int(row[5]), int(row[6])]
                cid = [int(row[8]), int(row[9])]
                tid = [int(row[11]), int(row[12])]
                rposes = pred_dict[key][rid[0] - sid : rid[0] + rid[1] - sid]
                cposes = pred_dict[key][cid[0] - sid : cid[0] + cid[1] - sid]
                tposes = pred_dict[key][tid[0] - sid : tid[0] + tid[1] - sid]
                
                if not (rposes.shape[0] > 0 and cposes.shape[0] > 0 and tposes.shape[0] > 0):
                    # filter out samples with segments < 1 frame long.
                    continue
                    
                ret = {'bar_outcome': int(row[3]), 'direction': row[4],
                       'runup_poses' : rposes,
                       'curve_poses' : cposes,
                       'takeoff_poses' : tposes,
                       'runup_labels': parse_labels(row[7]),
                       'curve_labels': parse_labels(row[10]),
                       'takeoff_labels': parse_labels(row[13]) }

                dataset[key] = ret

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
    dataset_pkl_file  (str) : Path to preprocessed pickle file.
    seg_name          (str) : Specify which segment to build dataset on, [runup, curve, takeoff]
    vocab_size        (int) : Specify how many most frequent labels to preserve. Will also be classifier dim.
    
    Outputs (__getitem__ returns):
    joints      : np.ndarray(T, 75)
    bar_outcome : int(1 or 0) 
    labels      : multi-hot vector in verbal case (e.g. if labels 2,5,3 are present among 6 classes, vector=[0,0,1,1,0,1])
    directions  : int(0 for left and 1 for right)
    names       : str(sequence name, for reference only, e.g. 'C0001')}
    """
    
    def __init__(self, dataset_pkl_file='data/dataset_with_labels.pkl', seg_name='runup', vocab_size=10):
        assert seg_name in ['runup', 'curve', 'takeoff']
        with open(dataset_pkl_file, 'rb') as handle:
            data = pickle.load(handle)
            
        self.joints = []# all seq files, num_videos * T(var) * 75
        self.bar_outcome = []
        self.labels = []
        self.names = []
        self.directions = []
        
        self.vocab_size = vocab_size
        
        # Temporary fake data generator. Will load and process from dataset later.
        # generate dictionary here
        count = []
        for key in data.keys():
            labels = data[key][seg_name+'_labels']
            for l in labels:
                count.append(l)
                
        counter = Counter(count)
        self.vocab2idx = defaultdict(lambda : vocab_size)
        idx = 0
        for k, v in counter.most_common(vocab_size):
            self.vocab2idx[k] = idx
            idx += 1
        
        self.idx2vocab = {v : k for k, v in self.vocab2idx.items()}
        for key in data.keys():
            self.joints.append(data[key][seg_name+'_poses'].reshape(-1, 75))  # T * 75
            self.names.append(key)
            self.directions.append(0 if data[key]['direction'] == 'left' else 1) # int
            
            self.bar_outcome.append(data[key]['bar_outcome'])
            self.labels.append(self.tokenize(data[key][seg_name+'_labels'])) 
            
        # prevent always choosing the same items for each fold, when doing kfold.
        temp = list(zip(self.joints, self.bar_outcome, self.labels, self.names, self.directions))
        random.shuffle(temp)
        self.joints, self.bar_outcome, self.labels, self.names, self.directions = zip(*temp)

    def tokenize(self, text):
        """
        Inputs:
        text (list) : list of terms (must be present in vocabulary), e.g. ["gin tonic", "kamikaze"]
        
        Outputs: 
        labels (np.ndarray) : multi-hot encoding of text, indicating the presence of each vocab item.  
        """
        multi_hot = np.zeros(self.vocab_size)
        for k, v in self.idx2vocab.items():
            if v in text:
                multi_hot[k] = 1
        return multi_hot

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.joints[idx], self.labels[idx], self.bar_outcome[idx], self.directions[idx], self.names[idx]

def collate_fn(batch):
    """
    Collate function.
    Processes the data after Frame.__getitem__, in order to pass them into dataloaders.
    
    Inputs: Batched data, [x, y, n] * batch_size.
    - joints (np.ndarray) : batch_size * T(var) * (25 * 3)
    - bar_outcome (int)   : batch_size * 1
    - labels (np.ndarray) : batch_size * vocab_size
    - directions (int)    : batch_size * 1
    - names (str)         : filename of sequence ('C0001' etc.), batch_size * 1
    
    Outputs: Stacks padded data into batches.
    - ret(dict): {
        - x_pad       (torch.Tensor)     : batch_size * max_T * 75 
        - x_lens      (torch.LongTensor) : batch_size * 1 (each T of x)
        - bar_outcome (torch.LongTensor) : batch_size * 1 (binary case)
        - labels      (torch.LongTensor) : batch_size * vocab_size (verbal case)
        - dir         (torch.LongTensor) : batch_size * 1
        - filename    (str)              : batch_size * 1
      }
    
    We have to pad the sequences with different lengths here & record their lengths respectively, for the dataloader & RNNs to work.
    """
    
    joints  = [item[0] for item in batch]
    labels = torch.LongTensor([item[1] for item in batch])
    bar_outcome = torch.LongTensor([item[2] for item in batch])
    directions = torch.LongTensor([item[3] for item in batch])
    names  = [item[4] for item in batch]

    x = [torch.Tensor(xx) for xx in joints]

    lens = torch.LongTensor([len(xx) for xx in joints])
    x_pad = pad_sequence([xx for xx in x], batch_first=True, padding_value=0)

    ret = {'x' : x_pad, 'x_lens' : lens, 'labels': labels, 'bar_outcome' : bar_outcome, 'dir': directions, 'filename' : names}
    return ret
