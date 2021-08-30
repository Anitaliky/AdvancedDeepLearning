import torch
import torch.nn as nn

import torchvision


class Model(nn.Module):
    def __init__(self,
                 backbone='resnet50',
                 num_classes=10,
                 feature_extraction_process=False, # true if we extract cifar10's features
                 feature_extraction_dataset=False # true if using modified (cifar10's extracted features) dataset
                ):
        super(Model, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.feature_extraction_dataset = feature_extraction_dataset
        self.model_type = vars(torchvision.models)[self.backbone]
        self.model = self.model_type(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        if feature_extraction_process:
            self.model.fc = nn.Identity()
        else:
            self.model.fc =  nn.Linear(self.model.fc.in_features, self.num_classes, bias=True)
        
            for p in self.model.fc.parameters():
                p.requires_grad = True
        
        
    def forward(self, x):
        if self.feature_extraction_dataset:
            x = self.model.fc(x)
        else:
            x = self.model(x)
        return x