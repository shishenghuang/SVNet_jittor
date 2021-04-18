
import numpy as np 
import jittor as jt 
from jittor import nn 
from jittor import Module 
from jittor import init 
#from jittor.contrib import concat , argmax_pool 


class SVNet(Module):
    def __init__(self, num_points_, num_k_, num_class_):
        super(SVNet, self).__init__()

        self.num_points = num_points_
        self.num_k = num_k_
        self.num_class = num_class_

        self.conv2d_x1 = nn.Conv2d(3,8 , 1)
        self.bn1 = nn.BatchNorm(8)
        self.conv2d_x2 = nn.Conv2d(8,16,1)
        self.bn2 = nn.BatchNorm(16)
        self.conv2d_x3 = nn.Conv2d(16,32,1)
        self.bn3 = nn.BatchNorm(32)
        self.conv2d_x4 = nn.Conv2d(32,64,1)
        self.bn4 = nn.BatchNorm(64)

        self.conv1d_x1 = nn.Conv1d(64*30,1024,1)
        self.bn5 = nn.BatchNorm(1024)
        self.conv1d_x2 = nn.Conv1d(1024,256,1)
        self.bn6 = nn.BatchNorm(256)
        self.conv1d_x3 = nn.Conv1d(256,128,1)
        self.bn7 = nn.BatchNorm(128)
        self.conv1d_x4 = nn.Conv1d(128,64,1)
        self.bn8= nn.BatchNorm(64)
        self.conv1d_x5 = nn.Conv1d(64,self.num_class,1)

        self.resize_l = nn.Resize([-1,64*30])
        self.softmax_k = nn.Softmax(self.num_class)

    
    def excute(self, points, features):

        x = nn.ReLU(self.bn1(self.conv2d_x1(points)))

        x = nn.ReLU(self.bn2(self.conv2d_x2(x)))

        x = nn.ReLU(self.bn3(self.conv2d_x3(x)))

        x = nn.ReLU(self.bn4(self.conv2d_x4(x)))

        y = x * features

        y = y.reshape([-1, 64*30])

        y = nn.ReLU(self.bn5(self.conv1d_x1(y)))

        y = nn.ReLU(self.bn6(self.conv1d_x2(y)))

        y = nn.ReLU(self.bn7(self.conv1d_x3(y)))

        y = nn.ReLU(self.bn8(self.conv1d_x4(y)))

        y = self.softmax_k(y)

        return y

    







