import os
import random
import numpy as np

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from net import SvhfNet, weights_init
from dataset import CustomDataset, data_transforms

root = 'pair_data'
root2 = 'test_pair_data'
train_dataset = CustomDataset(root, data_transforms['train'], train=True)
test_dataset = CustomDataset(root2, data_transforms['val'], train=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
net = SvhfNet().to(device)
net.apply(weights_init)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

num_epochs = 100

train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

for epoch in range(num_epochs):
    train_loss = 0
    train_acc = 0
    val_loss = 0
    val_acc = 0

    #train
    net.train()
    for i, (images, images2, sounds, labels) in enumerate(train_loader):
        images, images2, sounds, labels = images.to(device), images2.to(device), sounds.to(device, dtype=torch.float), labels.to(device)

        optimizer.zero_grad()
        outputs = net(images, images2, sounds)
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        train_acc += (outputs.max(1)[1] == labels).sum().item()
        loss.backward()
        optimizer.step()

    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_train_acc = train_acc / len(train_loader.dataset)

    #eval
    net.eval()
    with torch.no_grad():
        for images, images2, sounds, labels in test_loader:
            #view()での変換をしない
            images = images.to(device)
            images2 = images2.to(device)
            sounds = sounds.to(device, dtype=torch.float)
            labels = labels.to(device)
            outputs = net(images, images2, sounds)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_acc += (outputs.max(1)[1] == labels).sum().item()
    avg_val_loss = val_loss / len(test_loader.dataset)
    avg_val_acc = val_acc / len(test_loader.dataset)
    print('Epoch [{}/{}], Loss: {loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}'
            .format(epoch+1, num_epochs, i+1, loss=avg_train_loss, val_loss=avg_val_loss, val_acc=avg_val_acc))
    train_loss_list.append(avg_train_loss)
    train_acc_list.append(avg_train_acc)
    val_loss_list.append(avg_val_loss)
    val_acc_list.append(avg_val_acc)

torch.save(net.state_dict(), 'net.ckpt')