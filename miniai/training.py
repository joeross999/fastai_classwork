# AUTOGENERATED! DO NOT EDIT! File to edit: ../notebooks/04_minibatch.ipynb.

# %% auto 0
__all__ = ['accuracy', 'report', 'Dataset', 'fit', 'get_dls']

# %% ../notebooks/04_minibatch.ipynb 1
import pickle,gzip,math,os,time,shutil,torch,matplotlib as mpl,numpy as np,matplotlib.pyplot as plt
from pathlib import Path
from torch import tensor,nn
import torch.nn.functional as F
from functools import reduce
from torch import optim


# %% ../notebooks/04_minibatch.ipynb 6
def accuracy(out, yb): return (out.argmax(dim=1)==yb).float().mean()

# %% ../notebooks/04_minibatch.ipynb 7
def report(loss, preds, yb): print(f'{loss:.2f}, {accuracy(preds, yb):.2f}')

# %% ../notebooks/04_minibatch.ipynb 19
class Dataset():
    def __init__(self, x, y): self.x, self.y = x,y
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

# %% ../notebooks/04_minibatch.ipynb 36
def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb,yb in train_dl:
            loss = loss_func(model(xb), yb)
            loss.backward()
            opt.step()
            opt.zero_grad()

        model.eval()
        with torch.no_grad():
            tot_loss,tot_acc,count = 0.,0.,0
            for xb,yb in valid_dl:
                pred = model(xb)
                n = len(xb)
                count += n
                tot_loss += loss_func(pred,yb).item()*n
                tot_acc  += accuracy (pred,yb).item()*n
        print(epoch, tot_loss/count, tot_acc/count)
    return tot_loss/count, tot_acc/count

# %% ../notebooks/04_minibatch.ipynb 38
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, BatchSampler


# %% ../notebooks/04_minibatch.ipynb 40
def get_dls(train_ds, valid_ds, bs, **kwargs):
    return (DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
            DataLoader(valid_ds, batch_size=bs*2, shuffle=False, **kwargs))
