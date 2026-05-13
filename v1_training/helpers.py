from PIL import Image
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision import datasets

import pandas as pd

#Some global variables (hyperparameters)
#And helper functions


#####################################
#   PREPROCESSING TRANSFORMATIONS   #
#####################################

#Pixelstats for normalization (derived by measurement)
MEAN = [0.624, 0.520, 0.504]
STD = [0.242, 0.223, 0.231]

#Image size for model input
IMG_SIZE = (224, 224)   #A gpu-m nem bír többet :c
TILTED_SIZE = tuple(int(1.42*i) for i in IMG_SIZE)

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
def ComposeTrainingTransforms():
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
def ComposeRegularTransforms():
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


################
#   DATASETS   #
################

DS_PATH = "Data/medical data/skin_lesions_classification"   #Modified SLCD

def GetDataset(split, transform=ComposeRegularTransforms()):
    return datasets.ImageFolder(f"{DS_PATH}/{split}", transform=transform)

def GetTrainingDataset():
    return GetDataset("train", transform=ComposeTrainingTransforms())

def GetValDataset():
    return GetDataset("val", transform=ComposeRegularTransforms())

def GetTestDataset():
    return GetDataset("test", transform=ComposeRegularTransforms())



###################
#   DATALOADERS   #
###################

#In case I need something different
def GetDataloader(split, batch_size, num_workers):
    return DataLoader(GetDataset(split), batch_size=batch_size, shuffle=(split=="train"), num_workers=num_workers)

def GetTrainingDataloader(batch_size, num_workers):
    return DataLoader(GetTrainingDataset(), batch_size=batch_size, shuffle=True, num_workers=num_workers)

def GetValDataloader(batch_size, num_workers):
    return DataLoader(GetValDataset(), batch_size=batch_size, shuffle=False, num_workers=num_workers)

def GetTestDataloader(batch_size, num_workers):
    return DataLoader(GetTestDataset(), batch_size=batch_size, shuffle=False, num_workers=num_workers)



################
#   TRAINING   #
################

#TRAINING VARIABLES
MAX_EPOCHS = 50  #1 epoch kb 15p

def trainingEpoch(model, dataLoader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for image, labels in dataLoader:
        image = image.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(image)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size
    train_loss = running_loss / total
    train_acc = correct / total
    return train_loss, train_acc

@torch.no_grad()
def validate(model, dataLoader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    #confusion matrix
    classNames = dataLoader.dataset.classes
    CM = ComposeCM(classNames)

    for images, labels in dataLoader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)

        preds = logits.argmax(dim=1)

        for pred, label in zip(preds.cpu(), labels.cpu()):
            CM.loc[classNames[pred.item()], classNames[label.item()]] += 1

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    loss = running_loss / total
    acc = correct / total
    return loss, acc, CM




#############################
#   STATISTICS & METRICS    #
#############################

SAVE_PATH = "Models/"



def ComposeCM(classNames:list[str]) -> pd.DataFrame:
    #Convention: Column name is actual class, row name is predicted class
    return pd.DataFrame(columns=classNames, index=classNames).fillna(0)

#Flatten a CM into a single row so it can be stored in DF
def FlattenCM(cm:pd.DataFrame) -> dict[str, int]:
    #Format: cm_true_{name}_pred_{name}
    return {
        f"cm_true_{true}_pred_{pred}": int(cm.loc[pred, true])
        for pred in cm.index
        for true in cm.columns
    }

#Parse a flattened CM row from DF back
def ParseCM(CM:pd.DataFrame) -> pd.DataFrame:
    #Format: cm_true_{name}_pred_{name}
    ClassNames = sorted(list({ className.split("pred_")[1] for className in CM.keys() if "pred" in className} ))
    N_CM = ComposeCM(ClassNames)
    for key, val in CM.items():
        true, pred = key.split("cm_true_")[1].split("_pred_")
        N_CM.loc[pred, true] = int(val)
    return N_CM


#101:
# ComposeStats() & ComposeCM() to initialize
# DataFrame( FlattenCM() ) can be concated to stats
# Stats can be saved
# ParseCM( stats.iloc[row][3:] ) CM can be extracted from stats
    

#From a confusion matrix get class specific values
def PredictionsPerClass(CM, className):
    TP = CM[className][className]
    FN = CM[className].sum() - TP
    FP = CM.loc[className].sum() - TP
    TN = CM.values.sum() - (TP + FN + FP)
    return TP, FN, FP, TN

def ClassMetrics(TP, FN, FP, TN):
    accuracy = (TP + TN) / (TP + FN + FP + TN)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    F1_score = (2*precision*recall) / (precision + recall)
    return accuracy, precision, recall, F1_score
