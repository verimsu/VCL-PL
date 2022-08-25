import torch
import numpy as np
from utils.utils import AverageMeter, ProgressMeter
import time 

def pretraining(train_loader, model, criterion, c_criterion, optimizer, loss_optimizer, epoch):
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [losses],
        prefix="Epoch: [{}]".format(epoch))
    
    for k, batch in enumerate(train_loader):
        images = batch['image']
        images_waugmented = batch['image_waugmented']
        labels = batch['target']
        b, c, h, w = images.size()
        input_ = torch.cat([images.unsqueeze(1), images_waugmented.unsqueeze(1)], dim=1)
        input_ = input_.view(-1, c, h, w) 
        input_ = input_.cuda(non_blocking=True)
        ##############
        loss = 0.0
        output_ = model(input_)
        #labels = torch.arange(b).detach()
        for i, (output, mu, var) in enumerate(output_):
            output = output.view(b, 2, -1)
            embeddings = torch.cat([output[:,0,:], output[:,1,:]],dim=0)
            curr_labels = torch.cat([labels[:,i], labels[:,i]], dim=0).long()
            loss += criterion(embeddings, curr_labels)
        loss /= len(output_)
        ##############
        losses.update(loss.item())

        loss_optimizer.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_optimizer.step()

        if k % 50 == 0:
            progress.display(k)
