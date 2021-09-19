import os

import torch
import torch.nn as nn
import torchvision

from config import cfg



train_dataset = torchvision.datasets.CIFAR10(root=cfg.data_path, train=True, download=True)
test_dataset = torchvision.datasets.CIFAR10(root=cfg.data_path, train=False, download=True)
model0 = torchvision.models.resnet18(pretrained=True)
model1 = torchvision.models.resnet34(pretrained=True)
model2 = torchvision.models.resnet50(pretrained=True)
