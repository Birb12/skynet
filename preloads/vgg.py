import torchvision
import torch
import torchvision.transforms as transforms
import os
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

VGG = {
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG_net(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG['VGG19'])
        
        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
            )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        
        for x in architecture:
            if type(x) == int:
                out_channels = x
                
                layers += [nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                                     kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
                
        return nn.Sequential(*layers)


def vggTrain(epochs, optimizer, batchsize, learningrate, trainingdata):
    print(trainingdata)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    train_dataset = torchvision.datasets.ImageFolder(trainingdata, transform)

    loader = DataLoader(train_dataset, batchsize, shuffle=True)
    model = VGG_net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learningrate)
    criterion = nn.CrossEntropyLoss()
    least = 999

    if (optimizer == "Adam"):
        optimizer = torch.optim.Adam(model.parameters(), lr=learningrate)
    elif (optimizer == "SGD"):
        optimizer = torch.optim.SGD(model.parameters(), lr=learningrate)

    for epoch in range(epochs):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)
            loss = criterion(preds, labels)

            if (loss < least):
                save(model)
                least = loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print("Curr Epoch: " + str(epoch) + " | Loss: " + str(loss.item()))

def save(model):
    print("Highest accuracy reached, saved parameters")
    torch.save(model.state_dict(), "parameters/bestvgg.pth")
