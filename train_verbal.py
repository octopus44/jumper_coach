import argparse
import numpy as np

from dataset import Frames, collate_fn, kfold_split, match_labels
from models import SequenceAnalysis
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

def hamming_distance(a, b):
    a, b = a.cpu().detach().numpy(), b.cpu().detach().numpy()
    return np.sum([(a1 != b1) for (a1, b1) in zip(a, b)])

def train(model, dataset, epochs=200, lr=1e-3, n_splits=5):
    kfold_dataset = kfold_split(dataset, n_splits=n_splits)
    criterion = nn.BCELoss()
    model.cuda()
    
    val_acc_cv = 0
    train_acc_cv = 0
    #
    for f in range(n_splits):
        train_dataset, valid_dataset = kfold_dataset[f]
        trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
        validloader = DataLoader(valid_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
        
        model.reset()
        #optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.01)

        val_acc = 0
        train_acc = 0
        print(f"\nFold {f}")
        for e in range(epochs):
            
            train_loss = 0
            step = 0
            model.train()
            # Training loop
            for ret in trainloader:
                optimizer.zero_grad()
                x, x_lens, d, y = ret['x'].cuda(), ret['x_lens'], ret['dir'].cuda(), ret['labels'].cuda()
                probs = model(x, x_lens, d)
                loss = criterion(torch.sigmoid(probs), y.type(torch.FloatTensor).cuda())
                train_loss += loss.item()
                loss.backward(retain_graph=True)
                optimizer.step()
                step += 1

            # Validation loop: Accuracy is "open-ended" here.
            # Accuracy is defined as "element-wise" correctness here: 
            # Output logits are scaled between 0 and 1 with sigmoid (required for BCE loss, and output.shape = y.shape)
            # If sigmoid(x) > 0.5 then prediction = 1, else prediction = 0 (presence of each label).
            
            model.eval()
            errors = 0
            for ret in trainloader:
                x, x_lens, d, y = ret['x'].cuda(), ret['x_lens'], ret['dir'].cuda(), ret['labels'].cuda()
                probs = torch.sigmoid(model(x, x_lens, d)) # scales to between 0 and 1
                pred = torch.where(probs > 0.5, torch.ones_like(probs), torch.zeros_like(probs))
                #correct += torch.sum(pred == y)
                errors += hamming_distance(pred, y)
            train_acc = 1 - (errors / (len(train_dataset) * dataset.vocab_size))
            
            errors = 0
            for ret in validloader:
                x, x_lens, d, y = ret['x'].cuda(), ret['x_lens'], ret['dir'].cuda(), ret['labels'].cuda()
                probs = torch.sigmoid(model(x, x_lens, d))
                pred = torch.where(probs > 0.5, torch.ones_like(probs), torch.zeros_like(probs))
                errors += hamming_distance(pred, y)
            val_acc = 1 - (errors / (len(valid_dataset) * dataset.vocab_size))
            
            if (e+1) % (epochs // 5) == 0:
                print("Epoch: {}, Train Accuracy: {:.4f}, Valid Accuracy: {:.4f}".format(
                    e,train_acc , val_acc)) #train_loss / len(trainloader)

        val_acc_cv += val_acc
        train_acc_cv+= train_acc

    print(f"\nCross Validation Accuracy - Train:{train_acc_cv/n_splits}, Validation:{val_acc_cv/n_splits}")
    return model

if __name__ == "__main__":
    """
    Training code.
    $ python train.py [args]
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, help="specify parsed dataset location.", default='data/dataset_with_labels.pkl')
    parser.add_argument('--create_dataset', action='store_true', help='create dataset by merging files if specified.')
    parser.add_argument('--lr', type=float, default=1e-2, help='specify learning rate.')
    parser.add_argument('--epochs', type=int, default=100, help='specify number of epochs to train.')
    parser.add_argument('--n_splits', type=int, default=5, help='specify number of folds for K-fold cross validation.')
    parser.add_argument('--save_model_path', type=str, default='', help='specify model saving path.')
    args = parser.parse_args()
    
    print(args)
    if args.create_dataset:
        print("Merging dataset, saving to: {}".format(args.dataset_path))
        match_labels(trajectories='data/joint_trajectories_norm.pkl', annotations='data/labels.csv', output_file=args.dataset_path)
        
    dataset = Frames('data/dataset_with_labels.pkl', seg_name='takeoff')
    model = SequenceAnalysis(embed_dim=128, sliding_window_size=7, variant=0, vocab_size=dataset.vocab_size)
    model = train(model, dataset, epochs=args.epochs, lr=args.lr, n_splits=args.n_splits)
    if args.save_model_path:
        print("Saving model to: {}".format(args.save_model_path))
        torch.save(model.state_dict(), args.save_model_path)
