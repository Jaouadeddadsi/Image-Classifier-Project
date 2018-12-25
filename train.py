import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
from collections import OrderedDict
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
import argparse

### load data 
def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # Define transforms for the training, validation, and testing sets
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([transforms.RandomRotation(45),
                                           transforms.RandomResizedCrop(224), 
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           normalize,])
    valid_transforms = transforms.Compose([ transforms.Resize(256),
                                            transforms.RandomCrop(224), 
                                            transforms.ToTensor(),
                                            normalize,])
    #Load the datasets with ImageFolder
    trainset = datasets.ImageFolder(train_dir, transform=train_transforms)
    validset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    testset = datasets.ImageFolder(test_dir, transform=valid_transforms)
    return trainset, validset, testset

def build_model(arch = 'vgg19', hidden_units = 2960, lr = 0.001):
    models_list = {'vgg19': models.vgg19(pretrained=True), 'vgg11': models.vgg11(pretrained=True),'vgg13': models.vgg13(pretrained=True),\
                   'vgg16': models.vgg16(pretrained=True)}
    model = models_list[arch]
    for param in model.parameters():
        param.requires_grad = False
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                                        ('fc1', nn.Linear(25088, hidden_units,  bias=True)),
                                        ('Relu1', nn.ReLU()),
                                        ('Dropout1', nn.Dropout(p = 0.5)),
                                        ('fc2', nn.Linear(hidden_units, 102,  bias=True)),
                                        ('output', nn.LogSoftmax(dim=1))
                                        ]))
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    return model, criterion, optimizer

def validation(model, testloader, criterion, device = 'cuda'):
    test_loss = 0
    accuracy = 0
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy

def train_model( model, trainset, validset, validation,epochs = 8, device = 'cuda'): 
    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=64)
    running_loss = 0
    for e in range(epochs):
        model.train()
        model.to(device)
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                model.eval()
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validloader, criterion, device)
                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                            "Training Loss: {:.3f}.. ".format(running_loss/100),
                            "Test Loss: {:.3f}.. ".format(test_loss/len(validloader)),
                            "Test Accuracy: {:.3f}".format(accuracy/len(validloader)))
                
                running_loss = 0
                model.train()
    model.class_to_idx = trainset.class_to_idx 
    return model
def mk_checkpoint(model, arch = 'vgg19', hidden_units = 2960):
    checkpoint = {'arch': arch,
                  'hidden_units': hidden_units,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx
                    }
    return checkpoint


if __name__ == '__main__' :
    parser = argparse.ArgumentParser( description='Train a model')
    parser.add_argument('data', help = 'data directory')
    parser.add_argument('--save_dir', help = 'save directory')
    parser.add_argument('--arch', choices=[ 'vgg19', 'vgg11', 'vgg13', 'vgg16'] , help = 'model architecture')
    parser.add_argument('--learning_rate', help = 'the learning rate',  type=float)
    parser.add_argument('--epochs', help = 'number of epochs',  type=int)    
    parser.add_argument('--hidden_units', help = 'hidden units',  type=int)
    parser.add_argument('--gpu', help = 'execution on gpu', action="store_true")
    args = parser.parse_args()
   
    arch = args.arch if args.arch else 'vgg19'
    hidden_units = args.hidden_units if args.hidden_units  else  2960
    lr = args.learning_rate if args.learning_rate else  0.001
    epochs = args.epochs if args.epochs else 8
    device = 'cuda' if args.gpu else 'cpu'
    
    trainset, validset, testset = load_data(args.data)
    model, criterion, optimizer = build_model(arch = arch, hidden_units = hidden_units, lr = lr)  
    model = train_model(model, trainset, validset, validation, epochs, device = device)
    checkpoint = mk_checkpoint(model, arch = arch, hidden_units = hidden_units)
  
    if args.save_dir :
        torch.save(checkpoint, args.save_dir + 'checkpoint_1.pth')
    else :
        torch.save(checkpoint, 'checkpoint_1.pth')
        
    
    










