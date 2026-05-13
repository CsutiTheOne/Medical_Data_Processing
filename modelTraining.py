import os
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision import datasets

from consts import DS_PATH, MEAN, STD, IMG_SIZE, TILTED_SIZE
import metrics as m

"""
    This file is a function library to help
    model training easier
"""

#Configuration 

class Config:
    modelName = "name"
    purge     = False   #Start a new model from scratch
    training  = False   #to run training cycle
    
    batchSize  = 4 #Paralell training examples 
    workers    = 8 #CPU workers to load data to gpu
    learningRate = 1e-4
    weightDecay  = 1e-4

    training_epochs = 100

    def __init__(self, mname):
        self.modelName = mname
        self.getPaths()

    def getPaths(self):
        modelFolder = f"Models/{self.modelName}"
        plotsFolder = f"{modelFolder}/plots"
        self.plotsFolder = plotsFolder
        
        os.makedirs(plotsFolder, exist_ok=True)
        
        self.latestModelPath = f"{modelFolder}/latest.pth"  #weights after last training epoch
        self.bestValPath = f"{modelFolder}/best_val.pth"    #weights with highest validation score

        self.historyPath = f"{modelFolder}/history.csv"     #recorded stats during training
        self.statePath = f"{modelFolder}/state.pth"         #optimizer & current epoch





#####################################
#   PREPROCESSING TRANSFORMATIONS   #
#####################################

class CenterSquareCrop:
    #Relevant part is in the center
    #This crops the center in square ratio
    def __call__(self, img):
        w, h = img.size
        squareSide = min(w, h)
        l = (w - squareSide) // 2
        t = (h - squareSide) // 2
        r = l + squareSide
        b = t + squareSide
        return img.crop((l, t, r, b))

#Training includes rotation for augmentation
def composeTrainingTransforms():
    return transforms.Compose([
        CenterSquareCrop(),
        transforms.Resize(TILTED_SIZE),
        transforms.RandomRotation(degrees=180, fill=0, expand=True, interpolation=transforms.InterpolationMode.BILINEAR,),
        transforms.CenterCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

#General preprocess for val and test
def composeRegularTransforms():
    return transforms.Compose([
        CenterSquareCrop(),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

#Unnormalize image
#Usually for demo display
meanT = torch.tensor(STD)
stdT = torch.tensor(MEAN)
def UnnormalizeImage(img):
    return img * meanT + stdT



########################
#   DATALSET LOADERS   #
########################

def getDataset(split, transform=composeRegularTransforms()):
    return datasets.ImageFolder(f"{DS_PATH}/{split}", transform=transform)

def getDataLoaders(batchSize, workers):
    trainSet = getDataset("train", transform=composeTrainingTransforms())
    trainLoader = DataLoader(trainSet, shuffle=True, batch_size=batchSize, num_workers=workers)
    
    valSet = getDataset("val")
    valLoader = DataLoader(valSet, shuffle=False, batch_size=batchSize, num_workers=workers)
    
    testSet = getDataset("test")
    testLoader = DataLoader(testSet, shuffle=False, batch_size=batchSize, num_workers=workers)

    return trainLoader, valLoader, testLoader



################
#   TRAINING   #
################

@torch.no_grad()
def validate(model, dataLoader, criterion, device):
    """
        By iterating trough dataloader records 
        the performance of provided model
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    #confusion matrix
    classNames = dataLoader.dataset.classes
    CM = m.ComposeCM(classNames)

    for images, labels in dataLoader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images) #inference
        loss = criterion(logits, labels)

        running_loss += loss.item() * labels.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for pred, label in zip(preds.cpu(), labels.cpu()):
            CM.loc[classNames[pred.item()], classNames[label.item()]] += 1

    loss = running_loss / total
    acc = correct / total
    return loss, acc, CM


def trainingEpoch(model, dataLoader, criterion, optimizer, device):
    """
        Executes a single training epoch on the model
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataLoader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images) #inference
        loss = criterion(logits, labels)

        #updating model weights
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total
    return train_loss, train_acc


def trainingLoop(c:Config, model, dataLoader, criterion, optimizer, device):
    if not c.training:
        



