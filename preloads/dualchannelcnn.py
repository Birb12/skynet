import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import os
import torchvision
import torchvision.transforms.functional as TorchFunctional
import zipfile

change = {
    'curr': [32, 'M', 64, 'M', 128, 128, 'M', 256, 'M', 512, 'M'], # use this to customize settings later
}


class make_cnn(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(make_cnn, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(change['curr'])
        
        self.fcs = nn.Sequential(
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, num_classes)
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

        



class dualchannel(nn.Module):
    def __init__(self):
        super(dualchannel, self).__init__()
        self.ConvNetA = make_cnn()
        self.ConvNetB = make_cnn()
    def forward(self, x):
        x1 = self.ConvNetA.forward(x)
        x2 = self.ConvNetB.forward(x)

        x = torch.cat((x1, x2), dim=1)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
numepoch = 10
data_direction = os.getcwd()
transform = transforms.Compose([transforms.Resize((50, 50)), transforms.ToTensor()])

def find_datasetcnn(input1):
    print("Looking for datasets...")
    os.system("kaggle datasets list -s " + input1)

    inp = str(input("Download dataset (put name and author): "))
    os.system("kaggle datasets download -d " + inp)
    
    index = inp.find('/')
    strin = os.getcwd()

    if index != -1:
        inp = inp[index:]
    else:
        inp = inp

    try:
        assert os.path.isfile(strin + inp + ".zip")
        with zipfile.ZipFile(strin + inp + ".zip", 'r') as z:
            print("Extracting...")
            print("Depending on dataset size, this may take a while")
            z.extractall()
    except AssertionError:
        print(".zip not found, assuming extracted dataset")
    
    print("Directories found (pick extracted dataset): ")
    print([name for name in os.listdir(".") if os.path.isdir(os.getcwd())])
    finaldataset = input()



find_datasetcnn("breastcancer")


