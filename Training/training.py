from os import makedirs, path
from time import sleep
from collections import Counter
import gc

import torch
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_rows = None

import Training.library as lib
from Training.metrics import flattenCM

###################################
#   NOTEBOOK SPECIFIC RESOURCES   #
###################################

class Config:
    """
        To be used as a dataclass to configure
        the execution within a notebook
    """

    modelName = "name"
    purge     = False   #Start a new model from scratch
    training  = False   #to run training cycle
    loadBest  = False   #Load model with best validation score

    numClasses = 8 #Number of classes    
    batchSize  = 4 #Paralell training examples 
    workers    = 8 #CPU workers to load data to gpu
    learningRate = 1e-4
    weightDecay  = 1e-4

    trainingEpochs = 50
    gpuSleep        = 120 #Seconds to let gpu cool

    def __init__(self, mname):
        self.modelName = mname



class TrainingResources:
    """
        Class to group together every variable 
        of the model training process
    """
    epochs_done = 0
    history = None

    model = None
    
    criterion = optimizer = None
    train_loader = val_loader = test_loader = None

    def initPaths(self, modelName):
        modelFolder = f"Models/{modelName}"
        plotsFolder = f"{modelFolder}/plots"
        self.plotsFolder = plotsFolder
        
        makedirs(plotsFolder, exist_ok=True)
        
        self.latestModelPath = f"{modelFolder}/latest.pth"  #weights after last training epoch
        self.bestValPath = f"{modelFolder}/best_val.pth"    #weights with highest validation score
        self.historyPath = f"{modelFolder}/history.csv"     #recorded stats during training
        self.statePath = f"{modelFolder}/state.pth"         #optimizer & current epoch

    def __init__(self, c:Config, model, initTrainers:callable, device):
        self.initPaths(c.modelName)
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

        self.model = model.to(self.device)

        self.train_loader, self.val_loader, self.test_loader = lib.getDataLoaders(c.batchSize, c.workers, c.numClasses==2)

        self.criterion, self.optimizer = initTrainers(c, self.model, self.train_loader.dataset)
        # ensure criterion (loss weights) is on the same device
        try:
            self.criterion = self.criterion.to(self.device)
        except Exception:
            pass

    def load(self, c:Config):
        if not c.purge and path.exists(self.latestModelPath):
            #Note: this loads latest model by default
            #If config says to load bestVal, history is still 
            #for of the latest model
            map_loc = self.device
            self.model.load_state_dict(
                torch.load(self.bestValPath if c.loadBest else self.latestModelPath, map_location=map_loc)
            )
            
            state = torch.load(self.statePath, map_location=map_loc)
            self.optimizer.load_state_dict(state["optimizer"])
            self.epochs_done = state["epoch"]
            #load recorded stats too
            self.history = pd.read_csv(self.historyPath)
            print("Loaded state")
    
    def recordStat(self, statRow):
        if self.history is None:
            self.history = pd.DataFrame(columns=statRow.keys())
        self.history.loc[len(self.history)] = statRow

    def save(self):
        #Saving snapshot
        torch.save(self.model.state_dict(), self.latestModelPath)
        torch.save({
                "epoch": self.epochs_done,
                "optimizer": self.optimizer.state_dict()
            }, self.statePath)
        self.history.to_csv(self.historyPath, index=False)



#############################
#   LOSS FN AND OPTIMIZER   #
#############################

#For multi class classifier
def calcClassWeights(dataset):
    """
        Calculates the inverse frequenceis of
        each class to balance dataset
    """
    numClasses = len(dataset.classes)
    totalSamples = len(dataset)
    samplesPerClass = Counter(dataset.targets) 

    inverseFrequencies = [ totalSamples/(numClasses * samplesPerClass[i]) for i in range(numClasses) ]
    return inverseFrequencies

def initClassifierTrainers(c:Config, model, dataset):
    """
        Initializes loss function and optimizer
        Inverse freq calc for losssFn included
    """
    invFreqs = calcClassWeights(dataset)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(invFreqs))
    optimizer = torch.optim.AdamW(model.parameters(), lr=c.learningRate, weight_decay=c.weightDecay)

    return criterion, optimizer

#For binary classifier
def calcMelanomaWeights(dataset):
    totalSamples = len(dataset) #ok
    melanoma = 0
    nonMelanoma = 0
    for classIdx, count in Counter(dataset.targets).items():
        if dataset.classes[classIdx] == "Melanoma":
            melanoma += count
        else:
            nonMelanoma += count
    samples = [nonMelanoma, melanoma]
    numClasses = len(samples)
    return [ totalSamples/(numClasses * samples[i]) for i in range(numClasses) ] #inverse frequencies

def initMelanomaDetectorTrainers(c:Config, model, dataset):
    #Different optimizer and loss for melanoma detection
    invFreqs = calcMelanomaWeights(dataset)
    criterion = torch.nn.BCEWithLogitsLoss(weight=torch.tensor(invFreqs))
    optimizer = torch.optim.AdamW(model.parameters(), lr=c.learningRate, weight_decay=c.weightDecay)

    return criterion, optimizer


#############################
#   EXECUTION OF TRAINING   #
#############################

def trainingLoop(c:Config, tr:TrainingResources, stopAt = 0):
    if not c.training:
        print("Training turned off")
        return
    if c.loadBest:
        print("Probably best model is loaded, training would ruin it!")
        return

    #Free up memory
    gc.collect()
    torch.cuda.empty_cache()
    
    model = tr.model
    device = tr.device
    train_loader = tr.train_loader
    val_loader = tr.val_loader
    criterion = tr.criterion
    optimizer = tr.optimizer
    history = tr.history

    epochs = c.trainingEpochs
    if stopAt: epochs = stopAt

    bestTrainAcc = 0.0
    bestValAcc = 0.0
    if history is not None and len(history) > 0:
        bestTrainAcc = history["train_acc"].max()
        bestValAcc = history["val_acc"].max()

    for epoch in range(tr.epochs_done, epochs):
        print(f"Epoch {epoch+1}/{epochs} \t| ", end="")
        #Executing training and validation
        train_loss, train_acc = lib.trainingEpoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, CM = lib.validate(model, val_loader, criterion, device)

        print(
            f"train loss: {train_loss:.4f}, train acc: {train_acc:.4f} | "
            f"val loss: {val_loss:.4f}, val acc: {val_acc:.4f}"
        )
        tr.epochs_done += 1

        #Recording statistics
        statRow = {
            "epoch": epoch+1, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc
        }
        statRow.update(flattenCM(CM))
        tr.recordStat(statRow)
        tr.save()

        #saving best model
        if val_acc > bestValAcc:
            bestValAcc = val_acc
            torch.save(model.state_dict(), tr.bestValPath)
        
        if epoch == epochs-1: break
        sleep(c.gpuSleep) #Let GPU cool a little
    