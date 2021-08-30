import torch
import torch.nn as nn
from torch.utils.data import Dataset

from . import utils
from config import cfg

class CIFAR10_Resnet50(Dataset):
    def __init__(self, train=True):
        if train:
            self.data = torch.load(cfg.data_dir + '/exctracted_data/train.pt')
            self.targets = torch.load(cfg.data_dir + '/exctracted_data/train_targets.pt')
            
        else:
            self.data = torch.load(cfg.data_dir + '/exctracted_data/test.pt')
            self.targets = torch.load(cfg.data_dir + '/exctracted_data/test_targets.pt')
        
    def __len__(self):
        return self.data.size(1)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]