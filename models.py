import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class Model(nn.Module):
    def __init__(self, n_embeddings=100, sliding_window_size=7, n_conv_filters=60, n_rnn_hidden=20):
        """
        Define neural network.
        Inputs:
        - n_embeddings       : Size of feature vector for each joint / xyz.
        - sliding_window_size: Size of sliding window along time frames.
        - n_conv_filters     : Number of convolutional filters.
        - n_rnn_hidden       : Size of RNN hidden layer (neurons).
        
        Components:
        - self.mat        (nn.Linear) : Perform a linear transformation on all joints respectively, to get a larger feature size.
        - self.conv       (nn.Conv1d) : The "sliding window" operation, convolving along time frames -- with learnable weights.
        - self.rnn        (nn.RNN)    : Recurrent layer, outputs stacked hidden vectors at each time frame.
        - self.classifier (nn.Linear) : Based on the hidden vector on the last time frame, predicts whether the jump is successful.
        """
        super().__init__()
        self.sliding_window_size = sliding_window_size
        
        self.mat = nn.Linear(75, n_embeddings)
        self.conv = nn.Conv1d(in_channels=n_embeddings, 
                              out_channels=n_conv_filters, 
                              kernel_size=sliding_window_size)
        self.rnn = nn.RNN(input_size=n_conv_filters, 
                          hidden_size=n_rnn_hidden, 
                          bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(in_features=n_rnn_hidden * 2, out_features=2)

    def forward(self, x, x_lens):           
        """
        Forward pass of defined network.
        
        Inputs: 
        - x     (torch.Tensor(batch_size, max_length, 75)) : Input features. Padded to be the max length in each batch. 
        - x_len (torch.LongTensor(batch_size))             : Stores original lengths of each sample in batch, e.g.[122, 94, 130, 78]
        
        Outputs:
        - probs (torch.Tensor(batch_size, 2))              : Probability of each label (0 and 1).
        """
        
        x = self.mat(x)                                                    # bs * L * 100
        x = self.conv(x.permute(0,2,1))                                    # bs * n_filters * L' 
        x_lens = x_lens - self.sliding_window_size + 1                     # L' = L - k + 1. 
                                                                           # Shrinks because of the sliding window size.
        
        packed_x = pack_padded_sequence(x.permute(0,2,1), x_lens, batch_first=True, enforce_sorted=False)
        packed_out = self.rnn(packed_x)[0]
        out, out_lens = pad_packed_sequence(packed_out, batch_first=True)  # bs * L' * (n_hidden*2)
        out = out.permute(0, 2, 1)                                         # bs * (n_hidden*2) * L'

        st = []
        for inst, l in zip(out, out_lens):
            st.append(inst[:, l - 1])                                      # stack RNN output feature vectors & exclude padding
        st = torch.stack(st)                                               # bs * (n_hidden*2)
        
        #probs = F.softmax(self.classifier(st), dim=1)
        probs = self.classifier(st)                                        # outputs probability
        return probs

