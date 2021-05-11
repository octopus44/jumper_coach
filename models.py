import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from dataset import Frames, collate_fn

class PoseEmbedding(nn.Module):
    """
    Defines the feature embedding module.

    Variant 1:                     -----------add-----------      --------add--------
                                   |                       |      |                 |
    poses -> linear layer -> pose embed -> conv layer -> window embed -> RNN -> seq embed
                                 ^
    direction -> embedding layer | (add)
    (optional)
    
    Variant 2:                     ------------------------maxpool-------------------
                                   |                           |                    |
    poses -> linear layer -> pose embed -> conv layer -> window embed -> RNN -> seq embed
                                 ^
    direction -> embedding layer | (add)
    (optional)

    Components:
    - self.direction_embed : (optional) Returns a feature vector for each input index (0/1 for left/right).
    - self.frame_embed     : Perform a linear transformation on all joints respectively, to get desired feature size.
    - self.window_embed    : The "sliding window" operation, convolving along time frames -- with learnable weights.
    - self.sequence_embed  : Recurrent layer, outputs stacked hidden vectors at each time frame.
                             Default is set to single direction now (bidirectional=False).
    """
    def __init__(self, embed_dim, sliding_window_size, variant=0, use_direction=True):
        super().__init__()
        self.wsize = sliding_window_size
        self.variant=variant
        assert self.variant in range(3)
        
        self.use_direction = use_direction
        if use_direction:
            self.direction_embed = nn.Embedding(2, embed_dim)

        self.frame_embed = nn.Linear(75, embed_dim)
        self.window_embed = nn.Conv1d(in_channels=embed_dim, 
                                       out_channels=embed_dim, 
                                       kernel_size=sliding_window_size)
        self.sequence_embed = nn.RNN(input_size=embed_dim, 
                          hidden_size=embed_dim,
                          bidirectional=False,
                          batch_first=True)
        
    def reset(self):
        """
        For K-Fold cross validation.
        Try resetting model weights to avoid weight leakage.
        """
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                #print(f'Reset trainable parameters of layer = {layer}')
                layer.reset_parameters()

    def forward(self, x, lens, d=None):
        """
        Forward pass of embedding module.

        - d: direction embedding features. 
        - f: frame embedding features. (Fix: Use F.pad to pad zeros & maintain size.)
        - w: window embedding features.
        - s: sequence embedding features.
        """
        f = self.frame_embed(x)                                        # bs, L, embed_dim
        if self.use_direction:
            d = self.direction_embed(d)                                # bs, 1, embed_dim
            f += d.unsqueeze(1)                                        # incorporate direction features: broadcast along L dim.
        f_pad = F.pad(f, (0, 0, self.wsize//2, self.wsize//2))         # pad (w/2) zeros to head and tail along L dim.
        
        w = self.window_embed(f_pad.permute(0, 2, 1)).permute(0, 2, 1) # bs, L, embed_dim
        if self.variant == 1:
            w += f

        packed = pack_padded_sequence(w, lens, batch_first=True, enforce_sorted=False)
        packed = self.sequence_embed(packed)[0]
        s, lens = pad_packed_sequence(packed, batch_first=True)        # bs, L, embed_dim
        if self.variant == 1:
            s += w
        
        elif self.variant == 2:
            w = torch.cat([e.unsqueeze(3) for e in [f,w,s]], dim=3)
            w, _ = torch.max(w, dim=3)

        return w, lens

class Classifier(nn.Module):
    """
    Defines classifier composed of linear layers.
    Inputs: 
    layer_sizes (tuple): A combination of linear layer sizes (len >= 2).
                         e.g. (512, 64, 2) will define linear1(512, 64) and linear2(64, 2).
    """
    def __init__(self, *layer_sizes):
        super().__init__()
        layer_sizes = [*layer_sizes]
        assert len(layer_sizes) > 1
        
        self.linears = []
        for i in range(len(layer_sizes) - 1):
            self.linears.append(nn.Linear(in_features=layer_sizes[i], out_features=layer_sizes[i+1]))
        self.linears = nn.Sequential(*self.linears)
        self.activation = None
        
    def reset(self):
        """
        For K-Fold cross validation.
        Try resetting model weights to avoid weight leakage.
        """
        for layer in self.linears.children():
            if hasattr(layer, 'reset_parameters'):
                #print(f'Reset trainable parameters of layer = {layer}')
                layer.reset_parameters()

    def forward(self, x):
        x = self.linears(x)
        if self.activation:
            x = self.activation(x)
        return x

class SequenceAnalysis(nn.Module):
    """
    Defines full pipeline for each segment.

    Inputs:
    - embed_dim (int)           : Size of feature vectors. Keep consistent across all layers for stacking.
    - sliding_window_size (int) : Size of sliding window along time frames.
    - use_direction (bool)      : If true, add 'direction' (right/left) into features.
    - variant (int)             : Specify variant of embedding pipeline (0, 1, 2). Yet to be experimented.
    - vocab_size (int)          : If specified, will be the size of vocabulary (number of verbal suggestions).
                                  Otherwise, do classification for "good jump, bad jump".

    Components:
    - self.pose_embedding : PoseEmbedding() class. Module for feature embedding.
    - self.classifier     : Classifier() class. Module for predicting either verbal suggestion or jump success.
    """

    def __init__(self, embed_dim=128, sliding_window_size=7, variant=0, use_direction=True, vocab_size=None):
        super().__init__()
        self.pose_embedding  = PoseEmbedding(embed_dim=embed_dim, 
                                             sliding_window_size=sliding_window_size, 
                                             variant=variant,
                                             use_direction=use_direction)
        self.classifier      = Classifier(embed_dim, 2) if vocab_size is None \
                          else Classifier(embed_dim, vocab_size) 

    def reset(self):
        for module in self.children():
            if isinstance(module, PoseEmbedding):
                module.reset()
            elif isinstance(module, Classifier):
                module.reset()

    def forward(self, x, lens, d=None): 
        """
        Forward pass of defined network.
        
        Inputs: 
        - x     (torch.Tensor(batch_size, max_length, 75)) : Input features. Padded to be the max length in each batch. 
        - lens  (torch.LongTensor(batch_size))             : Stores original lengths of each sample in batch, e.g.[122, 94, 130, 78]
        - d     (torch.LongTensor(batch_size))             : 'direction' indicator for each input sequence. (0/1 for left/right)

        Outputs:
        - probs (torch.Tensor(batch_size, num_classes))    : Probability of each label.
        """

        x, lens = self.pose_embedding(x, lens, d) # bs, L, embed_dim
        x = x.permute(0, 2, 1)                    # bs, embed_dim, L

        st = []
        for inst, l in zip(x, lens):
            st.append(inst[:, l - 1])             # stack RNN output feature vectors & exclude padding
        st = torch.stack(st)                      # bs * embed_dim

        probs = self.classifier(st)               # outputs probability
        return probs

class JumpPrediction(nn.Module):
    def __init__(self, embed_dim=128, sliding_window_size=7, variant=0, use_direction=True, vocab_size=None):
        super().__init__()
        self.runup_embedding  = PoseEmbedding(embed_dim=embed_dim, 
                                             sliding_window_size=sliding_window_size, 
                                             variant=variant,
                                             use_direction=use_direction)
        self.curve_embedding  = PoseEmbedding(embed_dim=embed_dim, 
                                             sliding_window_size=sliding_window_size, 
                                             variant=variant,
                                             use_direction=use_direction)
        self.takeoff_embedding  = PoseEmbedding(embed_dim=embed_dim, 
                                             sliding_window_size=sliding_window_size, 
                                             variant=variant,
                                             use_direction=use_direction)
        self.classifier = Classifier(embed_dim * 3, 64, 2)
        self.classifier_r = Classifier(embed_dim, vocab_size)
        self.classifier_c = Classifier(embed_dim, vocab_size)
        self.classifier_t = Classifier(embed_dim, vocab_size)
        
    def reset(self):
        for module in self.children():
            if isinstance(module, PoseEmbedding):
                module.reset()
            elif isinstance(module, Classifier):
                module.reset()
        
    def forward(self, r, c, t, r_lens, c_lens, t_lens, d=None): 
        """
        Forward pass of defined network.
        
        Inputs: 
        - x     (torch.Tensor(batch_size, max_length, 75)) : Input features. Padded to be the max length in each batch. 
        - lens  (torch.LongTensor(batch_size))             : Stores original lengths of each sample in batch, e.g.[122, 94, 130, 78]
        - d     (torch.LongTensor(batch_size))             : 'direction' indicator for each input sequence. (0/1 for left/right)

        Outputs:
        - probs (torch.Tensor(batch_size, num_classes))    : Probability of each label.
        """
        r, r_lens = self.runup_embedding(r, r_lens, d) # bs, embed_dim, L
        c, c_lens = self.curve_embedding(c, c_lens, d)
        t, t_lens = self.takeoff_embedding(t, t_lens, d)
        
        st = [] # embedding vectors
        for x, lens in zip([r, c, t], [r_lens, c_lens, t_lens]):
            _st = []
            for inst, l in zip(x.permute(0, 2, 1), lens): # batch
                _st.append(inst[:, l - 1])             # stack RNN output feature vectors & exclude padding
            _st = torch.stack(_st)                      # bs * embed_dim
            st.append(_st)
            
        r_probs = self.classifier_r(st[0]) # bs * vocab_size
        c_probs = self.classifier_c(st[1])
        t_probs = self.classifier_t(st[2])
        seg_probs = torch.cat([r_probs, c_probs, t_probs], dim=1) # bs * (3*vocab_size)
        
        st = torch.cat(st, dim=1)
        probs = self.classifier(st) # bs * 2
        return probs, seg_probs
