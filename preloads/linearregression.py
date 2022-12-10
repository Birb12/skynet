import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np
import torch.nn as nn
import os

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out

def linearTrain(epochs, optimizer, batchsize, learningrate, trainingdata):
    print(trainingdata)
    x,y = np.loadtxt(trainingdata, unpack=True, delimiter=',')
    model = linearRegression(1, 1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if (optimizer == "Adam"):
        optimizer = torch.optim.Adam(model.parameters(), lr=learningrate)
    elif (optimizer == "SGD"):
        optimizer = torch.optim.SGD(model.parameters(), lr=learningrate)

    criterion = nn.MSELoss()

    least = 9999999

    print("Beginning training...")
    for epoch in range(epochs):
        x = torch.from_numpy(np.asarray(x))
        y = torch.from_numpy(np.asarray(y))
        x = x.float()
        y = y.float()
        inputs = x.to(device)
        labels = y.to(device)


        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        if (loss < least):
            save(model)

        optimizer.step()

        print('epoch {}: loss {}'.format(epoch, loss))

def save(model):
    print("Higher accuracy reached, model saved")
    torch.save(model.state_dict(), os.getcwd() + "/bestlinear.pth")

linearTrain(500, "Adam", 50, 0.1, "C:/Users/mined/Desktop/projects/skynet/SAT to GPA.csv")


