import numpy as np
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
def get_jaccard_score(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union
def get_accuracy_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)
def get_f1_score(y_true, y_pred,avg='macro'): # micro = accuracy
    return f1_score(y_true, y_pred,average=avg)