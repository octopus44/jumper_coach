import argparse
import numpy as np

from dataset import Frames, collate_fn, kfold_split, match_labels
from models import SequenceAnalysis, JumpPrediction
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

def hamming_distance(a, b):
    a, b = a.cpu().detach().numpy(), b.cpu().detach().numpy()
    return np.sum([(a1 != b1) for (a1, b1) in zip(a, b)])

def train(model, dataset, vocab_size, epochs, lr=1e-4, lr_seg=1e-3, lr_bar=1e-3, weight_decay=1e-2,  pos_weight=5.0, alpha=0.9, n_splits=5):
    """
    Training pipeline.
    
    Inputs:
    model                  (nn.Module) : model to be trained.
    dataset (torch.utils.data.Dataset) : dataset containing all training + validation samples.
    vocab_size                   (int) : keep n most frequent vocabulary terms for verbal label classification.
    epochs                       (int) : specify amount of epochs to train for.
    lr                         (float) : learning rate.
    lr_seg                     (float) : learning rate for segment classification layers.
    weight_decay               (float) : regularization penalty.
    pos_weight                 (float) : weighting for BCE with logits loss. Should approximately be (#neg_samples / #pos_samples).
    alpha                      (float) : loss = alpha * bar_prediction_loss + (1 - alpha) * verbal_label_loss.
    n_splits                     (int) : number of splits for K-Fold cross validation.  
    
    Outputs:
    model (nn.Module) : trained model.
    """
    kfold_dataset = kfold_split(dataset, n_splits=n_splits)
    
    #bin_criterion = nn.CrossEntropyLoss()
    bin_criterion = nn.BCEWithLogitsLoss()
    pos_weight = torch.ones(dataset.vocab_size * 3) * pos_weight
    seg_criterion = nn.BCEWithLogitsLoss(pos_weight.cuda())
    model.cuda()
    
    val_acc_seg_cv, train_acc_seg_cv = 0, 0
    val_acc_bin_cv, train_acc_bin_cv = 0, 0
    #
    for f in range(n_splits):
        train_dataset, valid_dataset = kfold_dataset[f]
        trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
        validloader = DataLoader(valid_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
        
        model.reset()
        seg_fc_params = [p for n,p in model.named_parameters() if n.startswith('classifier_')] # seg classifiers are 'classifier_r', etc.
        bar_fc_params = [p for n,p in model.named_parameters() if n.startswith('classifier.')]
        other_params = [p for n,p in model.named_parameters() if not n.startswith('classifier')]
        
        optimizer = torch.optim.AdamW([
            {'params': other_params},
            {'params': seg_fc_params, 'lr' : lr_seg},
            {'params': bar_fc_params, 'lr' : lr_bar}
        ], lr=lr)

        print(f"\nFold {f}")
        for e in range(epochs):
            step = 0
            model.train()
            # Training loop
            for ret in trainloader:
                optimizer.zero_grad()
                r, r_lens = ret['r_joints'].cuda(), ret['r_lens']
                c, c_lens = ret['c_joints'].cuda(), ret['c_lens']
                t, t_lens = ret['t_joints'].cuda(), ret['t_lens']
                d, y = ret['dir'].cuda(), ret['bar_outcome'].cuda()
                seg = torch.cat([ret['r_labels'], ret['c_labels'], ret['t_labels']],dim=1).type(torch.FloatTensor).cuda()
                probs, seg_probs = model(r, c, t, r_lens, c_lens, t_lens, d)
                loss1 = bin_criterion(probs.squeeze(1), y.type(torch.FloatTensor).cuda()) #squeeze
                loss2 = seg_criterion(seg_probs, seg) # bs*num_classes, bs*num_classes

                loss = alpha * loss1 + (1.0 - alpha) * loss2
                loss.backward(retain_graph=True)
                optimizer.step()
                step += 1
            
            # Validation loop
            model.eval()
            seg_accs = []
            bin_accs = []
            for loader, num_samples in zip([trainloader, validloader], [len(train_dataset), len(valid_dataset)]):
                bar_errors, seg_errors = 0, 0
                for ret in loader:
                    r, r_lens = ret['r_joints'].cuda(), ret['r_lens']
                    c, c_lens = ret['c_joints'].cuda(), ret['c_lens']
                    t, t_lens = ret['t_joints'].cuda(), ret['t_lens']
                    d, y = ret['dir'].cuda(), ret['bar_outcome'].cuda()
                    seg = torch.cat([ret['r_labels'], ret['c_labels'], ret['t_labels']],dim=1).type(torch.FloatTensor).cuda()
                
                    probs, seg_probs = model(r, c, t, r_lens, c_lens, t_lens, d) # scales to between 0 and 1
                    #correct += torch.sum(torch.argmax(probs, dim=1) == y)
                    pred = torch.where(torch.sigmoid(probs) > 0.5, torch.ones_like(probs), torch.zeros_like(probs)).squeeze(1) # bce
                    bar_errors += hamming_distance(pred, y)

                    pred = torch.where(torch.sigmoid(seg_probs) > 0.5, torch.ones_like(seg_probs), torch.zeros_like(seg_probs)) # bce
                    seg_errors += hamming_distance(pred, seg)
                seg_acc = 1 - (seg_errors / (num_samples * dataset.vocab_size * 3)) #
                bin_acc = 1 - (bar_errors / (num_samples)) 
                seg_accs.append(seg_acc)
                bin_accs.append(bin_acc)
                
            train_seg_acc, val_seg_acc = seg_accs
            train_bin_acc, val_bin_acc = bin_accs
            
            if (e+1) % (epochs // 5) == 0:
                print("Epoch: {}, [Bar] Train Acc: {:.4f}, Valid Acc: {:.4f}, [Seg] Train Acc: {:.4f}, Valid Acc: {:.4f}".format(
                    e,train_bin_acc , val_bin_acc, train_seg_acc , val_seg_acc))

        val_acc_seg_cv   += val_seg_acc
        train_acc_seg_cv += train_seg_acc
        val_acc_bin_cv   += val_bin_acc
        train_acc_bin_cv += train_bin_acc

    print("Cross Validation Accuracy - [Bar] Train: {:.4f}, Valid: {:.4f}".format(train_acc_bin_cv/n_splits, val_acc_bin_cv/n_splits))
    print("Cross Validation Accuracy - [Seg] Train: {:.4f}, Valid: {:.4f}".format(train_acc_seg_cv/n_splits, val_acc_seg_cv/n_splits))
    return model

if __name__ == "__main__":
    """
    Training code.
    $ python train.py [args]
    """
    
    parser = argparse.ArgumentParser()
    # Dataset arguments
    parser.add_argument('--dataset_path', type=str, help="specify parsed dataset location.", default='data/dataset_with_labels.pkl')
    parser.add_argument('--create_dataset', action='store_true', help='create dataset by merging files if specified.')
    parser.add_argument('--save_model_path', type=str, default='', help='specify model saving path.')
    
    # Model configurations
    parser.add_argument('--variant', type=int, default=0, help='specify variant of embedding method.')
    parser.add_argument('--embed_dim', type=int, default=128, help='specify dimension of feature vectors.')
    parser.add_argument('--sliding_window_size', type=int, default=3, help='specify size of 1D convolution.')
    parser.add_argument('--vocab_size', type=int, default=10, help='specify vocab size of dataset.')
    
    # Training configurations
    parser.add_argument('--epochs', type=int, default=100, help='specify number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-4, help='specify learning rate.')
    parser.add_argument('--lr_seg', type=float, default=1e-3, help='specify learning rate for segment classification layers.')
    parser.add_argument('--lr_bar', type=float, default=1e-3, help='specify learning rate for bar outcome classification layers.')
    
    # Other training hyperparameters
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='specify regularizer weight.')
    parser.add_argument('--pos_weight', type=float, default=5.0, help='specify weighting for bce loss.')
    parser.add_argument('--alpha', type=float, default=0.9, help='specify weighting between both loss functions.')
    parser.add_argument('--n_splits', type=int, default=5, help='specify number of folds for K-fold cross validation.')
    parser.add_argument('--seed', type=int, default=-1, help='specify random seed.')
    
    args = parser.parse_args()
    
    print(args)
    if args.create_dataset:
        print("Merging dataset, saving to: {}".format(args.dataset_path))
        match_labels(trajectories='data/joint_trajectories_norm.pkl', annotations='data/labels.csv', output_file=args.dataset_path)
        
    dataset = Frames(args.dataset_path, vocab_size=args.vocab_size, seed=args.seed if args.seed >= 0 else None)
    #print(dataset.neg_to_pos_ratio(field="labels"))
    #print(dataset.neg_to_pos_ratio(field="bar_outcome"))
    
    model = JumpPrediction(embed_dim=args.embed_dim, sliding_window_size=args.sliding_window_size, variant=args.variant, vocab_size=dataset.vocab_size)
    model = train(model, dataset, vocab_size=args.vocab_size, epochs=args.epochs, \
                  lr=args.lr, lr_seg=args.lr_seg, lr_bar=args.lr_bar, weight_decay=args.weight_decay, \
                  pos_weight=args.pos_weight, alpha=args.alpha, n_splits=args.n_splits)
    
    if args.save_model_path:
        print("Saving model to: {}".format(args.save_model_path))
        torch.save(model.state_dict(), args.save_model_path)
