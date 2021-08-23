import torch
import torch.nn as nn

import torchvision


class Model(nn.Module):
    def __init__(self,
                 backbone='resnet50',
                 num_classes=10):
        super(Model, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.model_type = vars(torchvision.models)[self.backbone]
        self.model = self.model_type(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.fc = nn.Sequential(nn.Linear(self.model.fc.out_features, self.model.fc.out_features, bias=True),
                                nn.ReLU(),
                                nn.Linear(self.model.fc.out_features, self.num_classes, bias=True))
        
        for p in self.model.fc.parameters():
            p.requires_grad = True
        
        
    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x