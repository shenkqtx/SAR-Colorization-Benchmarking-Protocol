import torch
import torch.nn as nn
from torch.nn import init
import functools

class Baseline_CNN_k9515_n64(nn.Module):
    def __init__(self):
        super(Baseline_CNN_k9515_n64, self).__init__()
        self.layer_1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, stride=1, padding=4),
                                     nn.ReLU()
                                     )
        self.layer_2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2),
                                     nn.ReLU()
                                     )
        self.layer_3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                     nn.ReLU()
                                     )
        self.layer_4 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, stride=1, padding=2),
                                     )

    def forward(self, input):
        feature = self.layer_1(input)
        feature = self.layer_2(feature)
        feature = self.layer_3(feature)
        output = self.layer_4(feature)
        return output

class Baseline_CNN_k9315_n64(nn.Module):
    def __init__(self):
        super(Baseline_CNN_k9315_n64, self).__init__()
        self.layer_1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, stride=1, padding=4),
                                     nn.ReLU()
                                     )
        self.layer_2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                                     nn.ReLU()
                                     )
        self.layer_3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
                                     nn.ReLU()
                                     )
        self.layer_4 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, stride=1, padding=2),
                                     )

    def forward(self, input):
        feature = self.layer_1(input)
        feature = self.layer_2(feature)
        feature = self.layer_3(feature)
        output = self.layer_4(feature)
        return output

class Baseline_CNN_k9115_n64(nn.Module):
    def __init__(self):
        super(Baseline_CNN_k9115_n64, self).__init__()
        self.layer_1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, stride=1, padding=4),
                                     nn.ReLU()
                                     )
        self.layer_2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0),
                                     nn.ReLU()
                                     )
        self.layer_3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
                                     nn.ReLU()
                                     )
        self.layer_4 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, stride=1, padding=2),
                                     )

    def forward(self, input):
        feature = self.layer_1(input)
        feature = self.layer_2(feature)
        feature = self.layer_3(feature)
        output = self.layer_4(feature)
        return output

class Baseline_CNN_k935_n32(nn.Module):
    def __init__(self):
        super(Baseline_CNN_k935_n32, self).__init__()
        self.layer_1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=9, stride=1, padding=4),
                                     nn.ReLU()
                                     )
        self.layer_2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
                                     nn.ReLU()
                                     )
        self.layer_3 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=3, kernel_size=5, stride=1, padding=2),
                                     )

    def forward(self, input):
        feature = self.layer_1(input)
        feature = self.layer_2(feature)
        output = self.layer_3(feature)
        return output

class Baseline_CNN_k935_n128(nn.Module):
    def __init__(self):
        super(Baseline_CNN_k935_n128, self).__init__()
        self.layer_1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=128, kernel_size=9, stride=1, padding=4),
                                     nn.ReLU()
                                     )
        self.layer_2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                                     nn.ReLU()
                                     )
        self.layer_3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, stride=1, padding=2),
                                     )

    def forward(self, input):
        feature = self.layer_1(input)
        feature = self.layer_2(feature)
        output = self.layer_3(feature)
        return output

class Baseline_CNN_k955(nn.Module):
    def __init__(self):
        super(Baseline_CNN_k955, self).__init__()
        self.layer_1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, stride=1, padding=4),
                                     nn.ReLU()
                                     )
        self.layer_2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2),
                                     nn.ReLU()
                                     )
        self.layer_3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, stride=1, padding=2),
                                     )

    def forward(self, input):
        feature = self.layer_1(input)
        feature = self.layer_2(feature)
        output = self.layer_3(feature)
        return output

class Baseline_CNN_k935(nn.Module):
    def __init__(self):
        super(Baseline_CNN_k935, self).__init__()
        self.layer_1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, stride=1, padding=4),
                                     nn.ReLU()
                                     )
        self.layer_2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                                     nn.ReLU()
                                     )
        self.layer_3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, stride=1, padding=2),
                                     )

    def forward(self, input):
        feature = self.layer_1(input)
        feature = self.layer_2(feature)
        output = self.layer_3(feature)
        return output

class Baseline_CNN_k915(nn.Module):
    def __init__(self):
        super(Baseline_CNN_k915, self).__init__()
        self.layer_1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, stride=1, padding=4),
                                     nn.ReLU()
                                     )
        self.layer_2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0),
                                     nn.ReLU()
                                     )
        self.layer_3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, stride=1, padding=2),
                                     )

    def forward(self, input):
        feature = self.layer_1(input)
        feature = self.layer_2(feature)
        output = self.layer_3(feature)
        return output

