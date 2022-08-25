"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
import faiss

from utils.config import create_config
from utils.common_config import get_criterion, get_model, get_train_dataset, get_train_dataloader, get_val_dataloader, \
                            get_train_transformations, get_val_transformations,  get_optimizer, adjust_learning_rate
from utils.memory import MemoryBank
from utils.train_utils import pretraining
from utils.utils import fill_memory_bank
from termcolor import colored

# Parser
parser = argparse.ArgumentParser(description='SimCLR')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
parser.add_argument("--batch_size", default=128, type=int, help="batch size per gpu")
parser.add_argument("--epochs", default=400, type=int, help="number of total epochs to run")
args = parser.parse_args()

def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]

    assert (preds_k == targets_k)  # one to one
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))
    return res

class FaissKMeans:
    def __init__(self, n_clusters, n_init=10, max_iter=300):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.kmeans = None
        self.cluster_centers_ = None
        self.inertia_ = None

    def fit(self, X):
        self.kmeans = faiss.Kmeans(d=X.shape[1],
                                   k=self.n_clusters,
                                   niter=self.max_iter,
                                   nredo=self.n_init)
        self.kmeans.train(X.astype(np.float32))
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]

    def predict(self, X):
        return self.kmeans.index.search(X.astype(np.float32), 1)[1]

def main():
    # Retrieve config file
    p = create_config(args.config_env, args.config_exp, args.batch_size, args.epochs)
    p['batch_size'] = args.batch_size
    p['epochs'] = args.epochs
    print(colored(p, 'red'))
    
    # Model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    print('Model is {}'.format(model.__class__.__name__))
    print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print(model)
    
    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True
       
    # Criterion
    ##############
    #### EDIT ####
    ##############
    print(colored('Retrieve criterion', 'blue'))
    criterion, c_criterion = get_criterion(p)
    print('Criterion is {}'.format(criterion.__class__.__name__))
    criterion = criterion.cuda()
    c_criterion = c_criterion.cuda()
    ##############
    
    # Optimizer and scheduler
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    loss_optimizer = torch.optim.SGD(criterion.parameters(), lr=0.01)
    print(optimizer)
 
    # Checkpoint
    start_epoch = 0
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    # Dataset    
    print(colored('Retrieve dataset', 'blue'))
    train_transforms = get_train_transformations(p)
    print('Train transforms:', train_transforms)
    
    # Memory Bank
    val_transforms = get_val_transformations(p)
    print('Val transforms:', val_transforms)
    print(colored('Build MemoryBank', 'blue'))
    base_dataset = get_train_dataset(p, val_transforms, to_augmented_dataset=True)
    base_dataloader = get_val_dataloader(p, base_dataset) 
    memory_bank_base = MemoryBank(len(base_dataset), 
                                p['model_kwargs']['features_dim'],
                                p['model_kwargs']['num_heads'], p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_base.cuda()
    
    # Mine the topk nearest neighbors at the very end (Train) 
    # These will be served as input to the SCAN loss.
    print(colored('Fill memory bank for mining the nearest neighbors (train) ...', 'blue'))
    fill_memory_bank(base_dataloader, model, memory_bank_base)
    topk = 1
    print('Mine the nearest neighbors (Top-%d)' %(topk)) 
    indices, acc = memory_bank_base.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on train set is %.2f' %(topk, 100*acc))
    np.save(p['topk_neighbors_train_path'], indices)   

    # Dataset
    train_dataset = get_train_dataset(p, train_transforms, to_augmented_dataset=True) # Split is for stl-10
    train_dataloader = get_train_dataloader(p, train_dataset)
    
    # Training
    print(colored('Starting main loop', 'blue'))
    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' %(epoch, p['epochs']), 'yellow'))
        print(colored('-'*15, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))
        
        # Train
        model.train()
        print('Train ...')
        pretraining(train_dataloader, model, criterion, c_criterion, optimizer, loss_optimizer, epoch)
        
        model.eval()
        with torch.no_grad():
            # Mine the topk nearest neighbors at the very end (Train) 
            print(colored('Fill memory bank for mining the nearest neighbors (train) ...', 'blue'))
            fill_memory_bank(base_dataloader, model, memory_bank_base)
            topk = 1
            print('Mine the nearest neighbors (Top-%d)' %(topk)) 
            indices, acc = memory_bank_base.mine_nearest_neighbors(topk)
            print('Accuracy of top-%d nearest neighbors on train set is %.2f' %(topk, 100*acc))         
            
            if epoch == 0:
                best_acc = acc

            if acc >= best_acc:
                # Checkpoint
                print('Checkpoint ...')
                torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 'epoch': epoch + 1}, os.path.join(p['pretext_dir'], 'finetuning_model.pth.tar'))
                best_acc = acc     
                print("Best Accuracy:", best_acc)
                np.save(p['topk_neighbors_train_path'], indices)         

if __name__ == '__main__':
    main()
