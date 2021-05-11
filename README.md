# jumper_coach
High Jump coach integrating computer vision, track analysis, pose estimation, and expert training systems.
## Installation
TODO: python library setup, etc. 
## Loading Pose Data
For information on how to load the dataset of 3D joint trajectories see [doc/DATASET.md](https://github.com/octopus44/jumper_coach/doc/DATASET.md).

## Training
`$ python train.py [args]`

```
args:
    variant             (int) : Specify embedding method (see models.py for details). 
    embed_dim           (int) : Specify dimension of embedding vectors.
    sliding_window_size (int) : Specify kernel size of 1D convolution.
    vocab_size          (int) : keep n most frequent vocabulary terms for verbal label classification.
    epochs              (int) : specify amount of epochs to train for.
    lr                (float) : learning rate.
    lr_seg            (float) : learning rate for segment classification layers.
    weight_decay      (float) : regularization penalty.
    pos_weight        (float) : weighting for BCE with logits loss. Should approximately be (#neg_samples / #pos_samples).
    alpha             (float) : loss = alpha * bar_prediction_loss + (1 - alpha) * verbal_label_loss.
    n_splits            (int) : number of splits for K-Fold cross validation.  
```
