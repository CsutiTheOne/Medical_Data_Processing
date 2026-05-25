import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

import Training.metrics as m

# Training loss and accuracy

def plotTraining(history, plotsFolder):
    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="Training Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(plotsFolder + "/training_loss")
    plt.show()

    plt.figure()
    plt.plot(history["epoch"], history["train_acc"], label="Training Accuracy")
    plt.plot(history["epoch"], history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(plotsFolder + "/training_acc")
    plt.show()


def plotBinaryTraining(history, plotsFolder):
    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="Training Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(plotsFolder + "/training_loss")
    plt.show()


# Confusion matrix and metrics
def plotConfusionMatrix(CM, plotsFolder, threshold=None):
    D_CM = CM.T.apply(pd.to_numeric)
    cm_norm = D_CM.div(D_CM.sum(axis=1), axis=0)
    disp = ConfusionMatrixDisplay(
        display_labels=cm_norm.index,
        confusion_matrix=cm_norm.values
    )
    disp.plot(values_format=".2f")
    title = f"Confusion Matrix"
    if not threshold is None: title += f" th={(threshold*100):.0f}%"
    plt.title(title)
    plt.xticks(rotation=90)
    plt.savefig(plotsFolder + "confusion_matrix.png")
    plt.show()


# Deriving metrics
def deriveMetrics(CM, plotsFolder):
    metrics = {}

    for className in CM.index:
        TP, FN, FP, TN = m.predictionsPerClass(CM, className)
        accuracy, precision, recall, f1 = m.classMetrics(TP, FN, FP, TN)
        metrics[className] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    metrics = pd.DataFrame(metrics).T
    metrics = metrics.mul(100).round(2)
    metrics.to_csv(plotsFolder + "/metrics.csv")
    return metrics



def bestMelanoma(TR):
    bestCM = None
    atEpoch = -1
    bestF1 = 0
    at = 0

    #Find best melanoma score
    for i in range(len(TR.history)):
        
        CM = m.parseCM(TR.history.iloc[i][5:])
        #melanomaScores
        TP, FN, FP, TN = m.predictionsPerClass(CM, "Melanoma")
        _, _, _, F1_score = m.classMetrics(TP, FN, FP, TN)
        if F1_score > bestF1:
            bestF1 = F1_score
            bestCM = CM
            atEpoch = TR.history.iloc[i]["epoch"]
            at = i
        
    print(f"Best Melanoma validation @ epoch {atEpoch}")
    plotConfusionMatrix(bestCM, "../.")
    return deriveMetrics(bestCM, "../.").loc["Melanoma"]



def plotGradCam(original, visualizations, gcPath):
    fig, axes = plt.subplots(
        1,
        len(visualizations) + 1,
        figsize=(4 * (len(visualizations) + 1), 4)
    )

    axes[0].imshow(original)
    axes[0].set_title("Input")
    axes[0].axis("off")

    for ax, (name, vis) in zip(axes[1:], visualizations.items()):
        ax.imshow(vis)
        ax.set_title(name)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(gcPath)
    plt.show()

