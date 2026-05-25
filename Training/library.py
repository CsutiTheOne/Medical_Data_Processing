import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from Training.consts import DS_PATH, MEAN, STD, IMG_SIZE, TILTED_SIZE
from Training.metrics import initCM

"""
    This file is a function library to help
    model training easier
"""


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
meanT = torch.tensor(MEAN).view(3, 1, 1)
stdT = torch.tensor(STD).view(3, 1, 1)
def UnnormalizeImage(img):
    img = img.clone()          # shape [C,H,W]
    img = img * stdT + meanT
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).cpu().numpy()
    return img



###########################
#   DATALSETS & LOADERS   #
###########################

class BinaryMelanomaDataset(datasets.ImageFolder):
    #This class wraps around the original ds
    #but turns every class other then melanoma to 0
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.originalClasses = self.classes
        self.classes = ["Non-Melanoma", "Melanoma"]

    def __getitem__(self, index):
        img, originalLabel = super().__getitem__(index)
        className = self.originalClasses[originalLabel]
        newLabel = 1.0 if className == "Melanoma" else 0.0
        return img, torch.tensor(newLabel, dtype=torch.float32)


#For multi class classifiers
#aka non-binary datasets
def getDataset(split, transform=composeRegularTransforms(), binary=False):
    ds = [datasets.ImageFolder, BinaryMelanomaDataset][int(binary)]
    return ds(f"{DS_PATH}/{split}", transform=transform)

def getClassifierDatasets():
    trainSet = getDataset("train", transform=composeTrainingTransforms())
    valSet = getDataset("val")
    testSet = getDataset("test")
    return trainSet, valSet, testSet

#For binary melanoma detector
def getBinaryDatasets():
    trainSet = getDataset("train", transform=composeTrainingTransforms(), binary=True)
    valSet = getDataset("val", binary=True)
    testSet = getDataset("test", binary=True)
    return trainSet, valSet, testSet


def getDataLoaders(batchSize, workers, binary=False):
    trainSet, valSet, testSet = getBinaryDatasets() if binary else getClassifierDatasets()
    
    trainLoader = DataLoader(trainSet, shuffle=True, batch_size=batchSize, num_workers=workers)
    valLoader = DataLoader(valSet, shuffle=False, batch_size=batchSize, num_workers=workers)
    testLoader = DataLoader(testSet, shuffle=False, batch_size=batchSize, num_workers=workers)

    return trainLoader, valLoader, testLoader

#Binary és non-binary közötti választás lehetne sokkal rövidebb de mostmár nem varjálom

##################################
#   FUNCTION USED FOR TRAINING   #
##################################

@torch.no_grad()
def validate(model, dataLoader, criterion, device, binaryThreshold=None):
    """
        By iterating trough dataloader records 
        the performance of provided model
    """
    model.to(device)
    criterion.to(device)
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    #confusion matrix
    classNames = dataLoader.dataset.classes
    CM = initCM(classNames)

    for images, labels in dataLoader:
        images = images.to(device)
        labels = labels.to(device)
        if binaryThreshold is not None:
            labels = labels.float().unsqueeze(1)

        logits = model(images) #inference
        loss = criterion(logits, labels)

        running_loss += loss.item() * labels.size(0)

        probs = torch.sigmoid(logits)
        preds = probs >= binaryThreshold if binaryThreshold else logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for pred, label in zip(preds.cpu(), labels.cpu()):
            CM.loc[classNames[int(pred.item())], classNames[int(label.item())]] += 1 #Ez valószínüleg fordítva gyűjt

    loss = running_loss / total
    acc = correct / total
    return loss, acc, CM


def trainingEpoch(model, dataLoader, criterion, optimizer, scheduler, device, binaryThreshold=None):
    """
        Executes a single training epoch on the model
    """
    model.to(device)
    criterion.to(device)
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataLoader:
        images = images.to(device)
        labels = labels.to(device)
        if binaryThreshold is not None:
            labels = labels.float().unsqueeze(1)

        optimizer.zero_grad()

        logits = model(images) #inference
        loss = criterion(logits, labels)

        #updating model weights
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits)
        preds = probs >= binaryThreshold if binaryThreshold is not None else logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total
    return train_loss, train_acc
