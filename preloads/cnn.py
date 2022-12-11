import torchvision
import torch
import torchvision.transforms as transforms
import os
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader


class ConvNetwork(nn.Module):
    def __init__(self):

        super(ConvNetwork, self).__init__()

        self.layer1 = nn.Conv2d(3, 6, 5)
        self.poollayer = nn.MaxPool2d(2, 2)
        self.layer2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(1296, 100)
        self.fc2 = nn.Linear(100, 20)
        self.fc3 = nn.Linear(20, 2)

    def forward(self, x):

        x = self.poollayer(F.relu(self.layer1(x)))
        x = self.poollayer(F.relu(self.layer2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def cnnTrain(epochs, optimizer, batchsize, learningrate, trainingdata):
    print(trainingdata)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.Resize((50, 50)), transforms.ToTensor()])
    train_dataset = torchvision.datasets.ImageFolder(trainingdata, transform)

    loader = DataLoader(train_dataset, batchsize, shuffle=True)
    model = ConvNetwork().to(device)
    
    if (optimizer == "Adam"):
        optimizer = torch.optim.Adam(model.parameters(), lr=learningrate)
    elif (optimizer == "SGD"):
        optimizer = torch.optim.SGD(model.parameters(), lr=learningrate)

    criterion = nn.CrossEntropyLoss()
    least = 999

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
    torch.save(model.state_dict(), "parameters/bestcnn.pth")


def cnnImplement(imagepath): # run one image through the model
    model = ConvNetwork()
    model.load_state_dict(torch.load(os.getcwd() + "/parameters/bestcnn.pth"))
    model.eval()

    image = Image.open(imagepath)
    transform = transforms.Compose([transforms.Resize((50, 50)), transforms.ToTensor()])
    image = transform(image).float()
    image = image.unsqueeze(0)

    output = model(image)
    _, predicted = torch.max(output.data, 1)
    return predicted.item()

cnnImplement("images.jpg")