import fileinput
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import zipfile

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def startup():
    currmodels = {"linearregression", "dualcnn", "cnn", "resnet50", "efficientnetb4"}
    print("Welcome to Skynet! Current models are listed below: ")
    
    for i in currmodels:
        print(i)
    
    curr_model = input("Pick a model to train: ")
    curr_dataset = input("What do you want this model to be trained on? ")
    find_datasetcnn(curr_dataset)

def customize(curr_model):
    print("Entering customization mode with: " + curr_model)

startup()