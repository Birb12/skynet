import fileinput
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import zipfile
from preloads.linearregression import linearRegression, linearTrain, linearImplement
from preloads.dualchannelcnn import dualchannel, dualTrain, dualImplement
from preloads.cnn import ConvNetwork, cnnTrain, cnnImplement
from preloads.vgg import VGG_net, vggTrain, vggImplement
from preloads.efficientnet import EfficientNet, efficientTrain, efficientImplement
import pandas as pd
from colorama import Fore

def find_dataset(input1, curr_model):
    print("Looking for datasets...")
    os.system("kaggle datasets list -s " + input1)

    inp = str(input("Download dataset (put name and author): "))
    os.system("kaggle datasets download -d " + inp)
        
    index = inp.find('/')
    strin = os.getcwd()

    if index != -1: # if found /, get rid of all characters beforehand
        inp = inp[index:]
    else:
        inp = inp
    
    global finaldataset

    try:
        assert os.path.isfile(strin + inp + ".zip") # assert .zip file is in directory
        with zipfile.ZipFile(strin + inp + ".zip", 'r') as z:
            print("Extracting...")
            print("Depending on dataset size, this may take a while")
            z.extractall()

        if curr_model == "linearregression":
            for file in os.listdir(os.getcwd()): # find files ending in .csv
                if file.endswith(".csv"):
                    finaldataset = os.path.join(os.getcwd(), file)
        else:
            for file in os.listdir(os.getcwd()): # look for folders that arent preloads
                d = os.path.join(os.getcwd(), file)
                if os.path.isdir(d):
                    if "preloads" in d:
                        continue
                    elif "parameters" in d:
                        continue
                    else:
                        finaldataset = d    
                        for file1 in os.listdir(d):
                            if file1 == "train": # a lot of datasets have the directory train, if not, just set dataset as is
                                finaldataset = os.path.join(d, os.path.basename(file1))

        return finaldataset
    except AssertionError: # if .zip file not found, catch exception and continue
        print(".zip not found, assuming pre-extracted dataset")
        
        print("Directories found (pick pre-extracted dataset): ")
        print([name for name in os.listdir(".") if os.path.isdir(os.getcwd())])
        finaldataset = input()
        return finaldataset

def asciibling():
    print("\n\n")
    print("""                                       ▄▄
                                    ,▄▓▓▓▓▄
                                  ╓▓▓▓▓▓▓▓▓▓▓,
                                 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                              ▄   ▓▓▓▓▓▓▓▓▓▓▓▓ ,▄
                           ,▄▓▓▓▓,  ▓▓▓▓▓▓▓▓ ,▓▓▓▓▄,
                         ,▓▓▓▓▓▓▓▓▓,  ▀▓▓▀  ╓▓▓▓▓▓▓▓▓▓,
                       ╓▓▓▓▓▓▓▓▓▓▓▓▓▓╖    ▄▓▓▓▓▓▓▓▓▓▓▓▓▓▄
                     ▄▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▄
                   ▄▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▄
                ,▄▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▄,
               ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓""")
    print("\n\n")

def startup(): # startup function
    currmodels = {"linearregression", "dualcnn", "cnn", "vgg19", "efficientnetb4"}
    asciibling()
    print("Welcome to Skynet! Current models are listed below: ")
    
    for i in currmodels:
        print(i)
    
    global curr_model
    curr_model = input("Pick a model to train: ")
    choice = input("Find a dataset, or upload your own (U, F): ")

    global finaldataset

    if choice == "F":
        input3 = input("Search for: ")
        finaldataset = find_dataset(input3, curr_model)
    elif choice == "U":
        input4 = input("Directory of dataset (must be in skynet cwd): ")
        finaldataset = input4

def customize(curr_model):
    global finaldataset
    print("Entering customization mode with: " + curr_model)
    epochs = int(input("# Epochs: "))
    optimizer = input("Optimizer (Adam, SGD): ")
    batchsize = int(input("Batch size: "))
    learningrate = float(input("Learning rate: "))
    trainingdata = os.path.join(os.getcwd(), finaldataset)

    customization = [epochs, optimizer, batchsize, learningrate, trainingdata]
    return customization

def skynetconsole():
    customization = [5, "Adam", 50, 0.1, os.getcwd()] # by default, if user chooses not to customize by themselves
    while True:
        userinput = input("skynet> ")

        if userinput == "start":
            startup()
        if userinput == "customize":
            customization = customize(curr_model)
        if userinput == "train":
            if curr_model == "linearregression":
                print("This is a linear regression model. All inputs are expected to be from a .csv")
                linearTrain(customization[0], customization[1], customization[2], customization[3], customization[4])
            elif curr_model == "dualcnn":
                dualTrain(customization[0], customization[1], customization[2], customization[3], customization[4])
            elif curr_model == "cnn":
                cnnTrain(customization[0], customization[1], customization[2], customization[3], customization[4])
            elif curr_model == "vgg19":
                vggTrain(customization[0], customization[1], customization[2], customization[3], customization[4])
            elif curr_model == "efficientnetb4":
                efficientTrain(customization[0], customization[1], customization[2], customization[3], customization[4])
        if userinput == "implement":
            if curr_model == "linearregression":
                x = int(input("Input LR 1: "))
                y = int(input("Input LR 2: "))
                li = [x, y]
                linearImplement(li)
            elif curr_model == "dualcnn":
                image = input("Input image path: ")
                print("Output: " + str(dualImplement(image)))
            elif curr_model == "cnn":
                image = input("Input image path: ")
                print("Output: " + str(cnnImplement(image)))
            elif curr_model == "vgg19":
                image = input("Input image path: ")
                print("Output: " + str(vggImplement(image)))
            elif curr_model == "efficientnetb4":
                image = input("Input image path: ")
                print("Output: " + str(efficientImplement(image)))

                


skynetconsole()

