import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

import Training.metrics as m

# Training loss and accuracy

#NOTE: Exporing to images should also be done here

def plotTraining(history, plotsFolder):
    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="Training Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig(plotsFolder + "/training_loss")

    plt.figure()
    plt.plot(history["epoch"], history["train_acc"], label="Training Accuracy")
    plt.plot(history["epoch"], history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig(plotsFolder + "/training_acc")


# Confusion matrix and metrics
def plotConfusionMatrix(CM, plotsFolder):
    CM = CM.apply(pd.to_numeric)
    cm_norm = CM.div(CM.sum(axis=1), axis=0)
    disp = ConfusionMatrixDisplay(
        display_labels=cm_norm.index,
        confusion_matrix=cm_norm.values
    )
    disp.plot(values_format=".2f")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=90)
    plt.show()
    plt.savefig(plotsFolder + "/confusion_matrix")


# Deriving metrics
def deriveMetrics(CM, plotsFolder):
    metrics = {}

    for className in CM.index:
        TP, FN, FP, TN = m.PredictionsPerClass(CM, className)
        accuracy, precision, recall, f1 = m.ClassMetrics(TP, FN, FP, TN)
        metrics[className] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    metrics = pd.DataFrame(metrics).T
    metrics = metrics.mul(100).round(4)
    metrics.to_csv(plotsFolder + "/metrics.csv")
    return metrics
