import os
import random
from tqdm import tqdm

import numpy as np
import pandas as pd

from PIL import ImageFilter  # , Image
import cv2

import torch
import torchvision

from .pytorch_utils.checkpoint import Checkpoint


# standard cifar10 stats
cifar10_mean = (0.5, 0.5, 0.5)
cifar10_std = (0.5, 0.5, 0.5)


class Config:
    """ a simple class for managing experiment setup """
    def __call__(self):
        return vars(self)

    def __repr__(self):
        return str(self())

    def __str__(self):
        return self.__repr__()
    
    
def accuracy_score(preds, targets):
    return float((preds == targets).astype(float).mean())
    

class Checkpoint(Checkpoint):
    def batch_pass(self,
                   device,
                   batch,
                   *args, **kwargs):
        results = {}
        
        imgs, targets = batch
        imgs, targets = imgs.to(device), targets.to(device)
        self.batch_size = imgs.size(0)

        out = self.model(imgs)
        preds = torch.argmax(out, dim=1)
        
        loss = self.criterion(out, targets)
        
        out = out.detach().cpu().numpy(),
        preds = preds.detach().cpu().numpy(),
        targets = targets.detach().cpu().numpy(),

        results = {
            'out': out,
            'preds': preds,
            'targets': targets
        }
        preds = preds[0]
        targest = targets[0]
    
        pbar_postfix = {
            'loss': float(loss.data),
            'score': self.score(preds, targets)
        }

        return loss, results, pbar_postfix

    def agg_results(self, results):
#         preds = np.concatenate(results['preds'])
#         targets = np.concatenate(results['targets'])

        preds = results['preds']
        targets = results['targets']
#         print(type(targets))
#         print(len(targets))
#         print(targets)
        targets = np.concatenate(tuple(targets))
        preds = np.concatenate(tuple([[t for t in y] for y in preds])).reshape(targets.shape)

        single_num_score = self.score(preds, targets)
        additional_metrics = {}

        return single_num_score, additional_metrics