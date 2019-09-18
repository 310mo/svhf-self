import os
import numpy as np
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

def weights_init(m):
    """
    ニューラルネットワークの重みを初期化する。作成したインスタンスに対しapplyメソッドで適用する
    :param m: ニューラルネットワークを構成する層
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:            # 畳み込み層の場合
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:        # 全結合層の場合
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:     # バッチノーマライゼーションの場合
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class SvhfNet(nn.Module):

    def __init__(self):
        super(SvhfNet, self).__init__()
        self.face_features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.face_fc1 = nn.Linear(7*7*256, 4096)
        self.face_bn1 = nn.BatchNorm1d(4096)
        self.face_relu1 = nn.ReLU(inplace=True)
        self.face_fc2 = nn.Linear(4096, 1024)

        self.sound_features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=7, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,2)),
        )
        self.sound_fc1 = nn.Conv1d(256, 4096, kernel_size=(10, 1))
        self.sound_bn1 = nn.BatchNorm2d(4096)
        self.sound_relu1 = nn.ReLU(inplace=True)
        self.sound_apoo1 = nn.AvgPool2d(kernel_size=(1,8), stride=1)
        self.sound_fc2 = nn.Linear(4096, 1024)

        self.all_fc1 = nn.Linear(3072, 1024)
        self.all_bn1 = nn.BatchNorm1d(1024)
        self.all_relu1 = nn.ReLU(inplace=True)
        self.all_fc2 = nn.Linear(1024, 512)
        self.all_bn2 = nn.BatchNorm1d(512)
        self.all_relu2 = nn.ReLU(inplace=True)
        self.all_fc3 = nn.Linear(512, 2)

    def forward(self, face1, face2, sound):

        face1 = self.face_features(face1)
        face1 = face1.view(face1.size(0), -1)
        face1 = self.face_fc1(face1)
        face1 = self.face_bn1(face1)
        face1 = self.face_relu1(face1)
        face1 = self.face_fc2(face1)

        face2 = self.face_features(face2)
        face2 = face2.view(face2.size(0), -1)
        face2 = self.face_fc1(face2)
        face2 = self.face_bn1(face2)
        face2 = self.face_relu1(face2)
        face2 = self.face_fc2(face2)

        sound = self.sound_features(sound)
        sound = self.sound_fc1(sound)
        sound = self.sound_bn1(sound)
        sound = self.sound_relu1(sound)
        sound = self.sound_apoo1(sound)
        sound = sound.view(sound.size(0), -1)
        sound = self.sound_fc2(sound)

        all_feature = torch.cat([face1, face2, sound], dim=-1)
        all_feature = self.all_fc1(all_feature)
        all_feature = self.all_bn1(all_feature)
        all_feature = self.all_relu1(all_feature)
        all_feature = self.all_fc2(all_feature)   
        all_feature = self.all_bn2(all_feature)
        all_feature = self.all_relu2(all_feature)     
        all_feature = self.all_fc3(all_feature)

        return all_feature