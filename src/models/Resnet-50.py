import torch
import torch.nn as nn
from torchvision import models

class ResNet50_GenreClassifier(nn.Module):
    def __init__(self):
        
        super().__init__()
        
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT )

        original_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        # assigining weights for this new channel as sum of old channels
        with torch.no_grad():
            self.resnet.conv1.weight[:] = original_conv.weight.sum(dim=1, keepdim=True)


        num_ftrs = self.resnet.fc.in_features
        
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.3), # Adding dropout for better regularization
            nn.Linear(num_ftrs, 10)
        )

    def forward(self, x):
        return self.resnet(x)