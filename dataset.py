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
    vocab_size        (int) : Specify how many most frequent labels to preserve. Will also be classifier dim.
    seed              (int) : Random seed.
    
    Outputs (__getitem__ returns):
    for [runup, curve, takeoff]:
        joints      : np.ndarray(T, 75)
        labels      : multi-hot vector (e.g. if labels 2,5,3 are present among 6 classes, vector=[0,0,1,1,0,1])
    bar_outcome : int(1 or 0) 
    directions  : int(0 for left and 1 for right)
    names       : str(sequence name, for reference only, e.g. 'C0001')}
    """
    
    def __init__(self, dataset_pkl_file='data/dataset_with_labels.pkl', vocab_size=10, seed=None):
        with open(dataset_pkl_file, 'rb') as handle:
            data = pickle.load(handle)
            
        self.r_joints, self.c_joints, self.t_joints = [], [], [] # all seq files, num_videos * T(var) * 75
        self.bar_outcome = []
        self.r_labels, self.c_labels, self.t_labels = [], [], []
        self.names = []
        self.directions = []
        
        self.vocab_size = vocab_size
        
        # Temporary fake data generator. Will load and process from dataset later.
        # generate dictionary here
        self.r_vocab2idx, self.c_vocab2idx, self.t_vocab2idx = {}, {}, {} # vocab to id
        for seg, mapping in zip(['runup', 'curve', 'takeoff'], [self.r_vocab2idx, self.c_vocab2idx, self.t_vocab2idx]):
            count = []
            for key in data.keys():
                labels = data[key][seg+'_labels']
                for l in labels:
                    count.append(l)
                
            counter = Counter(count)
            idx = 0
            for k, v in counter.most_common(vocab_size):
                mapping[k] = idx
                idx += 1
        
        for key in data.keys(): # all videos
            
            self.names.append(key)
            self.directions.append(0 if data[key]['direction'] == 'left' else 1) # int
            self.bar_outcome.append(data[key]['bar_outcome'])
            
            for seg, joints, labels, mapping in zip(['runup', 'curve', 'takeoff'], 
                                                     [self.r_joints, self.c_joints, self.t_joints],
                                                     [self.r_labels, self.c_labels, self.t_labels],
                                                     [self.r_vocab2idx, self.c_vocab2idx, self.t_vocab2idx]):
                
                joints.append(data[key][seg+'_poses'].reshape(-1, 75))  # T * 75
                labels.append(self.tokenize(data[key][seg+'_labels'], mapping=mapping)) 
            
        # prevent always choosing the same items for each fold, when doing kfold.
        temp = list(zip(self.r_joints, self.c_joints, self.t_joints, self.r_labels, self.c_labels, self.t_labels, \
                        self.bar_outcome, self.names, self.directions))
        if seed:
            random.seed(seed)
        random.shuffle(temp)
        self.r_joints, self.c_joints, self.t_joints, self.r_labels, self.c_labels, self.t_labels, \
        self.bar_outcome, self.names, self.directions = zip(*temp)

    def tokenize(self, text, mapping):
        """
        Inputs:
        text    (list) : list of terms (must be present in vocabulary), e.g. ["gin tonic", "kamikaze"].
        mapping (dict) : dictionary mapping vocabulary to index.
        
        Outputs: 
        labels (np.ndarray) : multi-hot encoding of text, indicating the presence of each vocab item.  
        """
        multi_hot = np.zeros(self.vocab_size)
        for v, k in mapping.items(): # descending order from most common
            if v in text:
                multi_hot[k] = 1
        return multi_hot


    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        return self.r_joints[idx], self.c_joints[idx], self.t_joints[idx], \
                self.r_labels[idx], self.c_labels[idx], self.t_labels[idx], \
                self.bar_outcome[idx], self.directions[idx], self.names[idx]

def collate_fn(batch):
    """
    Collate function.
    Processes the data after Frame.__getitem__, in order to pass them into dataloaders.
    
    Inputs: Batched data, [x, y, n] * batch_size.
    for [runup, curve, takeoff]:
        - joints (np.ndarray) : batch_size * T(var) * (25 * 3)
        - labels (np.ndarray) : batch_size * vocab_size
        
    - bar_outcome (int)   : batch_size * 1
    - directions (int)    : batch_size * 1
    - names (str)         : filename of sequence ('C0001' etc.), batch_size * 1
    
    Outputs: Stacks padded data into batches.
    - ret(dict): {
        for [runup, curve, takeoff]:
            - pad       (torch.Tensor)     : batch_size * max_T * 75 
            - lens      (torch.LongTensor) : batch_size * 1 (each T of x)
            - labels    (torch.LongTensor) : batch_size * vocab_size
            
        - bar_outcome (torch.LongTensor) : batch_size * 1 (binary)
        - dir         (torch.LongTensor) : batch_size * 1
        - filename    (str)              : batch_size * 1
      }
    
    We have to pad the sequences with different lengths here & record their lengths respectively, for the dataloader & RNNs to work.
    """
    
    r_joints  = [item[0] for item in batch]
    c_joints  = [item[1] for item in batch]
    t_joints  = [item[2] for item in batch]
    
    r_labels = torch.LongTensor([item[3] for item in batch])
    c_labels = torch.LongTensor([item[4] for item in batch])
    t_labels = torch.LongTensor([item[5] for item in batch])
    
    bar_outcome = torch.LongTensor([item[6] for item in batch])
    directions = torch.LongTensor([item[7] for item in batch])
    names  = [item[8] for item in batch]

    r_joints = [torch.Tensor(xx) for xx in r_joints]
    c_joints = [torch.Tensor(xx) for xx in c_joints]
    t_joints = [torch.Tensor(xx) for xx in t_joints]

    r_lens = torch.LongTensor([len(xx) for xx in r_joints])
    c_lens = torch.LongTensor([len(xx) for xx in c_joints])
    t_lens = torch.LongTensor([len(xx) for xx in t_joints])
    
    r_pad = pad_sequence([xx for xx in r_joints], batch_first=True, padding_value=0)
    c_pad = pad_sequence([xx for xx in c_joints], batch_first=True, padding_value=0)
    t_pad = pad_sequence([xx for xx in t_joints], batch_first=True, padding_value=0)

    ret = {'r_joints' : r_pad, 'c_joints' : c_pad, 't_joints' : t_pad, 'r_lens' : r_lens, 'c_lens' : c_lens, 't_lens' : t_lens,  \
           'r_labels': r_labels, 'c_labels': c_labels, 't_labels': t_labels,  \
           'bar_outcome' : bar_outcome, 'dir': directions, 'filename' : names
          }
    
    return ret
