from torch.utils.data import DataLoader
import torch
import torchvision
import torch.nn as nn

# 搭建网络
class SmallCifar(nn.Module):
    def __init__(self):
        super().__init__()
        self.module=nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=5, stride=1, padding=2),
            # nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            # nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            # nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Flatten(),
            # nn.ReLU(),
            # nn.Linear(1024, 10)
            )

    def forward(self,x):
        x = self.module(x)
        return x