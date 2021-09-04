# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 16:37:13 2021

@author: Lucas
"""


import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

confusion_matrix_df = pd.read_csv(r"C:\Users\Carlos\Documents\usp\patentClassifier\Patent_Classifier_Data\30mostImportant.tsv" , sep = "\t")
print(confusion_matrix_df.head(10))

y_actual = []
y_pred = []

confu_numpy = confusion_matrix_df.to_numpy()

i = 0
print(confu_numpy)
for column in confusion_matrix_df:
    if(column != 'Unnamed: 0'):
        for j in range(0,len(confu_numpy)):
            repetitions = confu_numpy[j][i]
            print(i," ",j)
            for c in range(0,int(repetitions)):
                y_actual.append(column)
                y_pred.append(confu_numpy[j][0])
    i+=1;

labels = []
for column in confusion_matrix_df:
    if(column != 'Unnamed: 0'):
        labels.append(column)
        

disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_actual, y_pred, labels=labels), display_labels=labels)
disp.plot()



