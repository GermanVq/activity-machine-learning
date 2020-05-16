# -*- coding: utf-8 -*-
"""
Created on Tue May 12 18:07:49 2020


                PRIME INDIANS DIABETES 

@author: German
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
import random
from warnings import simplefilter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

simplefilter(action='ignore', category=FutureWarning)


#Importing Dataset
url = 'diabetes.csv'
data = pd.read_csv(url)


#Define x and y
x = data.drop(['Outcome' ], axis=1)
y = data['Outcome'] # 1 -> no | 0 -> yes


#Splitting the Dataset into Training Set and Test Set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,stratify=y, random_state=0)


#The Training Set is unbalanced. And so become necessary to organize it.
pos_index = y_train[y_train.values == 1].index
neg_index = y_train[y_train.values == 0].index

if len(pos_index) > len(neg_index):
    higher = pos_index
    lower = neg_index
else:
    higher = neg_index
    lower = pos_index

random.seed(0)
higher = np.random.choice(higher, size=len(lower))
lower = np.asarray(lower)
new_indexes = np.concatenate((lower, higher))

x_train = x_train.loc[new_indexes]
y_train = y_train.loc[new_indexes]

#That implements the Transformer API to compute the mean and standard 
#deviation on a training set so as to be able to later reapply the 
#same transformation on the testing set.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train2 = pd.DataFrame(sc.fit_transform(x_train))
x_test2 = pd.DataFrame(sc.transform(x_test))
x_train2.columns = x_train.columns.values
x_test2.columns = x_test.columns.values
x_train2.index = x_train.index.values
x_test2.index = x_test.index.values
x_train = x_train2
x_test = x_test2

############################## Logistic Regression ##########################3
model = LogisticRegression(random_state = 0, penalty = 'l2')
model.fit(x_train, y_train)

# Predicting Test Set
acc_v = cross_val_score(estimator=model, X=x_train, y=y_train,cv=10)
acc_va = acc_v.mean()
acc_v.std()
y_pred = model.predict(x_test)
acc_t = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

probs_lr = model.predict_proba(x_test)
probs_lr = probs_lr[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, probs_lr)

#Confusion_matrix
cm_lg = confusion_matrix(y_test, y_pred)
print('-------confusion matrix LR--------')
print('', cm_lg) 
print('----------------------------------')

"""[Imprimir las métricas de precisión, recall y f1 de cada clase de cada
modelo]"""
#Classifecation report
print('----------------Classification report LR------------------')
print(classification_report(y_test,y_pred))

results = pd.DataFrame([['Logistic Regression (LR)', acc_va, acc_t, prec, rec, f1, auc]],
               columns = ['Model', 'Acc_Va', 'Acc_test', 'Precision', 'Recall', 'F1 Score', 'AUC'])
print('----------------------------------------------------------')



############################ K-Nearest Neighbors ##################################
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)

# Predicting Test Set
acc_v = cross_val_score(estimator=model, X=x_train, y=y_train,cv=10)
acc_va = acc_v.mean()
acc_v.std()
y_pred = model.predict(x_test)
acc_t = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

probs_k = model.predict_proba(x_test)
probs_k = probs_k[:, 1]
fpr_k, tpr_k, _ = roc_curve(y_test, probs_k)

#Confusion matrix
cm_k = confusion_matrix(y_test, y_pred)
print('--------Confusion matrix KKN--------')
print('', cm_k) 
print('------------------------------------')

"""[Imprimir las métricas de precisión, recall y f1 de cada clase de cada
modelo]"""
#Classifecation report
print('----------------Classification report KNN------------------')
print(classification_report(y_test,y_pred))
print('-----------------------------------------------------------')

model_results = pd.DataFrame([['K-Nearest Neighbors (KNN)', acc_va, acc_t, prec, rec, f1, auc]],
               columns = ['Model', 'Acc_Va', 'Acc_test', 'Precision', 'Recall', 'F1 Score', 'AUC'])

results = results.append(model_results, ignore_index = True)


###################################### SVM (Linear) ####################################
model = SVC(random_state = 0, kernel = 'linear', probability= True)
model.fit(x_train, y_train)

# Predicting Test Set
acc_v = cross_val_score(estimator=model, X=x_train, y=y_train,cv=10)
acc_va = acc_v.mean()
acc_v.std()
y_pred = model.predict(x_test)
acc_t = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

probs_svm = model.predict_proba(x_test)
probs_svm = probs_svm[:, 1]
fpr_svm, tpr_svm, _ = roc_curve(y_test, probs_svm)

#Confusion matrix
cm_svm = confusion_matrix(y_test, y_pred)
print('----confusion matrix SVM LINEAR-----')
print(':', cm_svm) 
print('------------------------------------')

"""[Imprimir las métricas de precisión, recall y f1 de cada clase de cada
modelo]"""
#Classifecation report
print('----------------Classification report SVM------------------')
print(classification_report(y_test,y_pred))
print('----------------------------------------------------------')

model_results = pd.DataFrame([['SVM (Linear)', acc_va, acc_t, prec, rec, f1, auc]],
               columns = ['Model', 'Acc_Va', 'Acc_test', 'Precision', 'Recall', 'F1 Score', 'AUC'])

results = results.append(model_results, ignore_index = True)


###################################### Decision Tree ##################################
model = DecisionTreeClassifier(criterion='entropy', random_state=0)
model.fit(x_train, y_train)

#Predicting Test Set
acc_v = cross_val_score(estimator=model, X=x_train, y=y_train,cv=10)
acc_va = acc_v.mean()
acc_v.std()
y_pred = model.predict(x_test)
acc_t = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

probs_dt = model.predict_proba(x_test)
probs_dt = probs_dt[:, 1]
fpr_dt, tpr_dt, _ = roc_curve(y_test, probs_dt)

#confusion matrix
cm_dt = confusion_matrix(y_test, y_pred)
print('-------confusion matrix DTREE------')
print('', cm_dt) 
print('-----------------------------------')

"""[Imprimir las métricas de precisión, recall y f1 de cada clase de cada
modelo]"""
#Classifecation report
print('----------------Classification report DT------------------')
print(classification_report(y_test,y_pred))
print('----------------------------------------------------------')

model_results = pd.DataFrame([['Decision tree (Dtree)', acc_va, acc_t, prec, rec, f1, auc]],
               columns = ['Model', 'Acc_Va', 'Acc_test', 'Precision', 'Recall', 'F1 Score', 'AUC'])

results = results.append(model_results, ignore_index = True)

#################################### Naive Bayes #################################
model = GaussianNB()
model.fit(x_train, y_train)

# Predicting Test Set
acc_v = cross_val_score(estimator=model, X=x_train, y=y_train,cv=10)
acc_va = acc_v.mean()
acc_v.std()
y_pred = model.predict(x_test)
acc_t = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

probs_nv = model.predict_proba(x_test)
probs_nv = probs_nv[:, 1]
fpr_nv, tpr_nv, _ = roc_curve(y_test, probs_nv)

#confusion matix 
cm_nb = confusion_matrix(y_test, y_pred)
print('--------confusion matrix NB---------')
print(':', cm_nb) 
print('------------------------------------')

"""[Imprimir las métricas de precisión, recall y f1 de cada clase de cada
modelo]"""
#Classifecation report
print('----------------Classification report NB------------------')
print(classification_report(y_test,y_pred))
print('----------------------------------------------------------')

model_results = pd.DataFrame([['Naive Bayes (Gauss)', acc_va, acc_t, prec, rec, f1, auc]],
               columns = ['Model', 'Acc_Va', 'Acc_test', 'Precision', 'Recall', 'F1 Score', 'AUC'])

results = results.append(model_results, ignore_index = True)

#####################################################################################

"""[Realizar usando la librería Pandas una tabla que resuma los resultados
de las métricas de cada modelo. En las filas debe ir el nombre del
modelo empleado y en las columnas los siguientes datos (ordenar de
mayor a menor por la métrica AUC):]"""
    
results_organize = results.sort_values(by=['AUC'],ascending=False)
print('-------------------------------------- RESULTS------------------------------------')
print('',results_organize)
print('----------------------------------------------------------------------------------')

#####################################################################################

""""[Imprimir las 5 matrices de confusión en un solo gráfico, empleando el
mapa de calor de la librería Seaborn]"""
## Evaluating Results
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(data=cm, annot=True)
plt.show()

"""[Mostrar las 5 curvas de ROC en el mismo gráfico]"""
##ROC curve
fpr_nv, tpr_nv, _ = roc_curve(y_test, probs_nv)
fpr_dt, tpr_dt, _ = roc_curve(y_test, probs_dt)
fpr_svm, tpr_svm, _ = roc_curve(y_test, probs_svm)
fpr_k, tpr_k, _ = roc_curve(y_test, probs_k)
fpr_lr, tpr_lr, _ = roc_curve(y_test, probs_lr)

plt.plot(fpr_nv, tpr_nv, color = 'orange', label = 'NV')
plt.plot(fpr_dt, tpr_dt, color = 'red', label = 'DT')
plt.plot(fpr_svm, tpr_svm, color = 'purple', label = 'SVM')
plt.plot(fpr_k, tpr_k, color = 'green', label = 'K')
plt.plot(fpr_lr, tpr_lr, color = 'black', label = 'LR')

plt.plot([0, 1],[0, 1], color = 'darkblue', linestyle = '--')
plt.show()