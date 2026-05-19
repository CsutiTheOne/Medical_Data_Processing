from pandas import DataFrame

#############################
#   STATISTICS & METRICS    #
#############################

def initCM(classNames:list[str]) -> DataFrame:
    #Convention: Column name is actual class, row name is predicted class
    return DataFrame(columns=classNames, index=classNames).fillna(0)

#Flatten a CM into a single row so it can be stored in DF
def flattenCM(cm:DataFrame) -> dict[str, int]:
    #Format: cm_true_{name}_pred_{name}
    return {
        f"cm_true_{true}_pred_{pred}": int(cm.loc[pred, true])
        for pred in cm.index
        for true in cm.columns
    }

#Parse a flattened CM row from DF back
def parseCM(CM:DataFrame) -> DataFrame:
    #Format: cm_true_{name}_pred_{name}
    ClassNames = sorted(list({ className.split("pred_")[1] for className in CM.keys() if "pred" in className} ))
    N_CM = initCM(ClassNames)
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
def predictionsPerClass(CM, className):
    TP = CM[className][className]
    FN = CM[className].sum() - TP
    FP = CM.loc[className].sum() - TP
    TN = CM.values.sum() - (TP + FN + FP)
    return TP, FN, FP, TN

def classMetrics(TP, FN, FP, TN):
    try:
        accuracy = (TP + TN) / (TP + FN + FP + TN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1_score = (2*precision*recall) / (precision + recall)
        return accuracy, precision, recall, F1_score
    except:
        return 0,0,0,0
