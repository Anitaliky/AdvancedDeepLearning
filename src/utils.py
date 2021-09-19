import os
import random
from tqdm import tqdm

import numpy as np
import pandas as pd
import random

from PIL import ImageFilter  # , Image
import cv2

import torch
import torchvision
import torchvision.transforms.functional as F

from typing import List

from .pytorch_utils.checkpoint import Checkpoint


# cifar10 stats
cifar10_mean = torch.tensor([0.4914, 0.4822, 0.4465])
cifar10_std = torch.tensor([0.2470, 0.2435, 0.2616])



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


class MyCheckpoint(Checkpoint):
    def batch_pass(self,
                   device,
                   batch,
                   *args, **kwargs):
        X = [b.to(device) for b in batch[:-1]]
        y = batch[-1].to(device)
        
        self.batch_size = y.shape[0]

        out = self.model(*X)
        loss = self.criterion(out, y)

        results = {
            'preds': out.detach().cpu().argmax(dim=1).numpy(),
            'trues': y.detach().cpu().clone().numpy()
            }

        pbar_postfix = {
            'loss': float(loss.data),
            'score': self.score(results['preds'], results['trues']),
        }

        return loss, results, pbar_postfix

    def agg_results(self, results):
        preds = np.concatenate(results['preds'])
        trues = np.concatenate(results['trues'])

        single_num_score = self.score(trues, preds)
        additional_metrics = {}

        return single_num_score, additional_metrics


class RotateAngle:
    """Rotate by one of the given angles."""

    def __init__(self, angles: List[float]=(0., ), **kwargs):
        self.angles = angles
        self.kwargs = kwargs

    def __call__(self, img):
        angle = random.choice(self.angles)
        return F.rotate(img=img, angle=angle, **self.kwargs)
    
    def __repr__(self):
        return f'RotateAngle(angles={self.angles}, {self.kwargs})'



