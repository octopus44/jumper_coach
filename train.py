import argparse

from dataset import Frames, collate_fn, kfold_split, match_data
from models import Model

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

def train(model, dataset, epochs=100, lr=1e-2, n_splits=5):
    kfold_dataset = kfold_split(dataset, n_splits=n_splits)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.cuda()
    
    for e in range(epochs):
        correct = 0
        train_loss = 0
        step = 0
        
        for f in range(n_splits):
            train_dataset, valid_dataset = kfold_dataset[f]
            trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
            validloader = DataLoader(valid_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
            model.train()
            # Training loop
            for ret in trainloader:
                optimizer.zero_grad()
                x, x_lens, y = ret['x'].cuda(), ret['x_lens'], ret['y'].cuda()
                probs = model(x, x_lens)
                loss = criterion(probs, y)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                step += 1
        
            # Validation loop
            model.eval()
            for ret in validloader:
                x, x_lens, y = ret['x'].cuda(), ret['x_lens'], ret['y'].cuda()
                probs = model(x, x_lens)
                correct += torch.sum(torch.argmax(probs, dim=1) == y)
            
        if (e+1) % 10 == 0 and f == n_splits - 1:
            print("Epoch: {}, Train Loss: {:.4f}, Valid Accuracy: {:.4f}".format(
                e, train_loss / step, correct / len(dataset)))
            
    return model

if __name__ == "__main__":
    """
    Training code.
    $ python train.py [args]
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, help="specify parsed dataset location.", default='data/dataset.pkl')
    parser.add_argument('--create_dataset', action='store_true', help='create dataset by merging files if specified.')
    parser.add_argument('--lr', type=float, default=1e-2, help='specify learning rate.')
    parser.add_argument('--epochs', type=int, default=100, help='specify number of epochs to train.')
    parser.add_argument('--n_splits', type=int, default=5, help='specify number of folds for K-fold cross validation.')
    parser.add_argument('--save_model_path', type=str, default='', help='specify model saving path.')
    args = parser.parse_args()
    
    print(args)
    if args.create_dataset:
        print("Merging dataset, saving to: {}".format(args.dataset_path))
        match_data(trajectories='data/joint_trajectories.pkl', annotations='data/annotations.csv', output_file=args.dataset_path)
        
    dataset = Frames(args.dataset_path)
    model = Model()
    model = train(model, dataset, epochs=args.epochs, lr=args.lr, n_splits=args.n_splits)
    if args.save_model_path:
        print("Saving model to: {}".format(args.save_model_path))
        torch.save(model.state_dict(), args.save_model_path)
