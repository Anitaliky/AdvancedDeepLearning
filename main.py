#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %config Completer.use_jedi = False
# %load_ext autoreload
# %autoreload 2


# In[2]:


import os
import random
from tqdm import tqdm

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision

from src import utils
from src import pytorch_utils as ptu
from config import cfg

import warnings
warnings.filterwarnings("ignore")


# In[3]:


# cfg.tqdm_bar = True
# cfg.prints = 'display'


# In[4]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device', device)


# In[5]:


transforms = torchvision.transforms.Compose([
    utils.RotateAngle(angles=cfg.angles),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(utils.cifar10_mean, utils.cifar10_std),
])


# In[7]:


# def extract_feats(model, loader, shuffle=False, tqdm_bar=False):
#     dataset_imgs = []
#     dataset_targets = []
#     model.eval()
#     with torch.no_grad():
#         if tqdm_bar:
#             pbar = tqdm(loader)
#         else:
#             pbar = loader
#         for imgs, targets in pbar:
#             embeds = model(imgs)
#             dataset_imgs.append(embeds.detach().cpu())
#             dataset_targets.append(targets)

#     dataset = torch.utils.data.TensorDataset(torch.cat(dataset_imgs), torch.cat(dataset_targets))
# #     data_loader = torch.utils.data.DataLoader(dataset,
# #                                               batch_size=loader.batch_size,
# #                                               num_workers=loader.num_workers,
# #                                               shuffle=shuffle,
# #                                               drop_last=loader.drop_last)

#     return dataset

# feature_extraction_model = vars(torchvision.models)[cfg.backbone](pretrained=True)
# for p in feature_extraction_model.parameters():
#     p.requires_grad = False
# feature_extraction_model.fc = nn.Identity()
# ptu.params(feature_extraction_model)

# train_loader = extract_feats(feature_extraction_model, train_loader, shuffle=True, tqdm_bar=cfg.tqdm_bar)
# test_loader = extract_feats(feature_extraction_model, test_loader, shuffle=False, tqdm_bar=cfg.tqdm_bar)

# torch.save(train_loader, os.path.join(cfg.data_path, 'train_dataset.pth'))
# torch.save(test_loader, os.path.join(cfg.data_path, 'test_dataset.pth'))


# In[6]:


train_dataset = torchvision.datasets.CIFAR10(root=cfg.data_path, train=True, transform=transforms)  # download=True ,
test_dataset = torchvision.datasets.CIFAR10(root=cfg.data_path, train=False, transform=transforms)  # download=True ,

if cfg.feature_extraction:
    train_loader = torch.utils.data.DataLoader(torch.load(os.path.join(cfg.data_path, 'train_dataset.pth')),
                                               batch_size=cfg.bs,
                                               num_workers=cfg.num_workers,
                                               shuffle=True,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(torch.load(os.path.join(cfg.data_path, 'test_dataset.pth')),
                                              batch_size=cfg.bs,
                                              num_workers=cfg.num_workers,
                                              shuffle=False,
                                              drop_last=True)
else:
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.bs,
                                               num_workers=cfg.num_workers,
                                               shuffle=True,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=cfg.bs,
                                              num_workers=cfg.num_workers,
                                              shuffle=False,
                                              drop_last=True)

# In[9]:


# cfg.load = None
# cfg.save = False
# cfg.optimizer = 'adam'


# In[10]:


print(f'Loads {cfg.version}')
if cfg.load is not None and os.path.exists(os.path.join(cfg.models_dir, cfg.version, ptu.naming_scheme(cfg.version, epoch=model_epoch)) + '.pth'):
    checkpoint = ptu.load_model(device, version=cfg.version, models_dir=cfg.models_dir, epoch=model_epoch)
    if cfg.prints == 'display':
        display(checkpoint.log.sort_index(ascending=False).head(20))
    elif cfg.prints == 'print':
        print(checkpoint.log.sort_index(ascending=False).head(20))
else:
    if cfg.feature_extraction:
        model = nn.Linear(train_loader.dataset.tensors[0].shape[1], len(train_dataset.classes), bias=cfg.bias)
    else:
        model = vars(torchvision.models)[cfg.backbone](pretrained=True)
        for p in model.parameters():
            p.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes), bias=cfg.bias)
    model.to(device)
    
    if cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                    lr=cfg.lr,
                                    momentum=cfg.optimizer_momentum,
                                    weight_decay=cfg.wd)
    elif cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                    lr=cfg.lr,
                                    weight_decay=cfg.wd)
    else:
        raise NotImplementedError
    
    criterion = nn.CrossEntropyLoss().to(device)
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                              T_max=cfg.epochs,
                                                              eta_min=cfg.min_lr) if cfg.cos else None
    
    checkpoint = utils.MyCheckpoint(version=cfg.version,
                                    model=model,
                                    optimizer=optimizer,
                                    lr_scheduler=lr_scheduler,
                                    criterion=criterion,
                                    score=utils.accuracy_score,
                                    models_dir=cfg.models_dir,
                                    best_policy=cfg.best_policy,
                                    save=cfg.save,
                                   )
ptu.params(checkpoint.model)


# In[ ]:


checkpoint.train(train_loader=train_loader,
                 val_loader=test_loader,
                 train_epochs=int(max(0, cfg.epochs - checkpoint.get_log())),
                 optimizer_params=cfg.optimizer_params,
                 prints=cfg.prints,
                 epochs_save=cfg.epochs_save,
                 epochs_evaluate_train=cfg.epochs_evaluate_train,
                 epochs_evaluate_validation=cfg.epochs_evaluate_validation,
                 max_iterations_train=cfg.max_iterations,
                 max_iterations_val=cfg.max_iterations,
                 device=device,
                 tqdm_bar=cfg.tqdm_bar,
                 save=cfg.save,
                 save_log=cfg.save_log,
                )


# In[ ]:


# checkpoint.summarize()


# In[ ]:


# results_log = pd.DataFrame(columns=['model', 'augment', 'class', 'angle', 'loss', 'score'])


# In[ ]:


# # all classes
# for angle in range(0, 91, 10):
#     print(f'Angle {angle}')
#     test_loader = dl.test_loader(data_dir=cfg.data_dir,
#                                  batch_size=cfg.bs,
#                                  augment=True,
#                                  angles=[angle])
#     loss, score, results = checkpoint.evaluate(loader=test_loader,
#                                                device=device,
#                                                tqdm_bar=True)
#     df = df.append({'model': 'base', 'augment': 'rotation', 'class': 'all', 'angle': angle, 'loss': loss, 'score': score},
#                    ignore_index=True)


# In[ ]:


# df


# In[ ]:


# # by class by angle classes
# for class_name in utils.classDict.keys():
#     for angle in range(0, 91, 10):
#         print(f'Class {class_name}, Angle {angle}')
#         test_loader = dl.test_loader(data_dir=cfg.data_dir,
#                                      batch_size=cfg.bs,
#                                      augment=True,
#                                      angles=[angle],
#                                      class_name=class_name
#                                     )
#         loss, score, results = checkpoint.evaluate(loader=test_loader,
#                                                    device=device,
#                                                    tqdm_bar=True)
#         df = df.append({'model': 'base', 'augment': 'rotation', 'class': class_name, 'angle': angle, 'loss': loss, 'score': score},
#                        ignore_index=True)


# In[ ]:


# df


# In[ ]:


# df.to_csv('log.csv', index=False)


# In[ ]:


# fig, axes = plt.subplots(figsize=(20,10),
#                          nrows=2, ncols=6)
# for (val, group), ax in zip(df.groupby('class'), axes.flatten()):
#     group.plot(x='angle', y='loss', kind='bar', ax=ax, title=val, ylim=(0, 7))


# In[ ]:


# # df.groupby('class').plot.bar(x='angle', y='score', ylim=(0, 1), subplots=True)
# fig, axes = plt.subplots(figsize=(20,10),
#                          nrows=2, ncols=6)
# for (val, group), ax in zip(df.groupby('class'), axes.flatten()):
#     group.plot(x='angle', y='score', kind='bar', ax=ax, title=val, ylim=(0, 1))


# ## RandomRotation Model
# ##### Differ from base model in training with rotations at random angles from list [0, 10, ..., 90]

# In[ ]:


# cfg.model_type = 'randomRotation'
# model_epoch = 'best'
# if cfg.load is not None and os.path.exists(os.path.join(cfg.models_dir, cfg.version, ptu.naming_scheme(cfg.version, epoch=model_epoch)) + '.pth'):
#     print(f'Loads {cfg.version}')
#     checkpoint = ptu.load_model(device, version=cfg.version, models_dir=cfg.models_dir, epoch=model_epoch)
#     if cfg.prints == 'display':
#         display(checkpoint.log.sort_index(ascending=False).head(20))
#     elif cfg.prints == 'print':
#         print(checkpoint.log.sort_index(ascending=False).head(20))
# else:
#     model = Model(backbone=cfg.backbone, num_classes=cfg.num_classes)
#     model.to(device)
    
#     if cfg.optimizer == 'sgd':
#         optimizer = torch.optim.SGD(model.model.fc.parameters(),
#                                     lr=cfg.lr,
#                                     momentum=cfg.optimizer_momentum,
#                                     weight_decay=cfg.wd)
#     else:
#         optimizer = torch.optim.Adam(model.model.fc.parameters(),
#                                     lr=cfg.lr,
#                                     weight_decay=cfg.wd)
    
#     criterion = nn.CrossEntropyLoss().to(device)
    
#     lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
#                                                               T_max=cfg.epochs,
#                                                               eta_min=cfg.min_lr)
    
#     checkpoint = utils.Checkpoint(version=cfg.version,
#                                   model=model,
#                                   optimizer=optimizer,
#                                   lr_scheduler=lr_scheduler,
#                                   criterion=criterion,
#                                   score=utils.accuracy_score,
#                                   models_dir=cfg.models_dir,
#                                   best_policy=cfg.best_policy,
#                                   save=cfg.save,
#                                  )


# In[ ]:


# train_loader, train_eval_loader = dl.train_eval_loaders_cifar10(data_dir=cfg.data_dir, batch_size=cfg.bs,
#                                                                 random_rotation=True, angles=list(range(0, 91, 10)))
# test_loader = dl.test_loader(data_dir=cfg.data_dir, batch_size=cfg.bs)


# In[ ]:


# checkpoint.optimizer = torch.optim.SGD(checkpoint.model.model.fc.parameters(),
#                                        lr=1e-3,  #cfg.lr,
#                                        momentum=cfg.optimizer_momentum,
#                                        weight_decay=cfg.wd)
# checkpoint.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(checkpoint.optimizer,
#                                                                      T_max=cfg.epochs,
#                                                                      eta_min=cfg.min_lr)


# In[ ]:


# checkpoint.train(train_loader=train_loader,
#                  train_eval_loader=train_eval_loader,
#                  val_loader=test_loader,
#                  train_epochs=int(max(0, cfg.epochs - checkpoint.get_log())),
#                  optimizer_params=cfg.optimizer_params,
#                  prints=cfg.prints,
#                  epochs_save=cfg.epochs_save,
#                  epochs_evaluate_train=cfg.epochs_evaluate_train,
#                  epochs_evaluate_validation=cfg.epochs_evaluate_validation,
#                  max_iterations_train=cfg.max_iterations,
#                  max_iterations_val=cfg.max_iterations,
#                  device=device,
#                  tqdm_bar=cfg.tqdm_bar,
#                  save=cfg.save,
#                  save_log=cfg.save_log,
#                 )


# In[ ]:


# checkpoint.summarize()


# In[ ]:


# # all classes
# for angle in range(0, 91, 10):
#     print(f'Angle {angle}')
#     test_loader = dl.test_loader(data_dir=cfg.data_dir,
#                                  batch_size=cfg.bs,
#                                  augment=True,
#                                  angles=[angle])
#     loss, score, results = checkpoint.evaluate(loader=test_loader,
#                                                device=device,
#                                                tqdm_bar=True)
#     df = df.append({'model': 'randomRotation', 'augment': 'rotation', 'class': 'all', 'angle': angle, 'loss': loss, 'score': score},
#                    ignore_index=True)


# In[ ]:


# df


# In[ ]:


# # by class by angle classes
# for class_name in utils.classDict.keys():
#     for angle in range(0, 91, 10):
#         print(f'Class {class_name}, Angle {angle}')
#         test_loader = dl.test_loader(data_dir=cfg.data_dir,
#                                      batch_size=cfg.bs,
#                                      augment=True,
#                                      angles=[angle],
#                                      class_name=class_name
#                                     )
#         loss, score, results = checkpoint.evaluate(loader=test_loader,
#                                                    device=device,
#                                                    tqdm_bar=True)
#         df = df.append({'model': 'randomRotation', 'augment': 'rotation', 'class': class_name, 'angle': angle, 'loss': loss, 'score': score},
#                        ignore_index=True)


# In[ ]:


# df


# In[ ]:


# df.to_csv('log.csv', index=False)


# In[ ]:


# fig, axes = plt.subplots(figsize=(20,10),
#                          nrows=2, ncols=6)
# for (val, group), ax in zip(df[df['model'] == 'randomRotation'].groupby('class'), axes.flatten()):
#     group.plot(x='angle', y='loss', kind='bar', ax=ax, title=val, ylim=(0, 7))


# In[ ]:


# # df.groupby('class').plot.bar(x='angle', y='score', ylim=(0, 1), subplots=True)
# fig, axes = plt.subplots(figsize=(20,10),
#                          nrows=2, ncols=6)
# for (val, group), ax in zip(df[df['model'] == 'randomRotation'].groupby('class'), axes.flatten()):
#     group.plot(x='angle', y='score', kind='bar', ax=ax, title=val, ylim=(0, 1))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




