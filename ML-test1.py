#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 18:23:36 2017

@author: charleslaidi
"""

import numpy as np
import os
import re
import glob
import pandas as pd
from sklearn import svm
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold, permutation_test_score
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn import grid_search
from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile, f_classif
import random
from sklearn.preprocessing import Imputer
from sklearn import svm
from sklearn import cross_validation
from sklearn import datasets
from sklearn.metrics import recall_score
from sklearn import svm, metrics, linear_model
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import  make_scorer,accuracy_score,recall_score,precision_score
from collections import Counter
from pandas import DataFrame
from pandas import Series
import patsy
import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import statsmodels.api as sm
from __future__ import print_function
from collections import Counter
from scipy import stats
import scipy


# =============================================================================
# IMPORTATION DU FICHIER CSV
# =============================================================================

# chemin vers la base
BASE_PATH = "/Users/charleslaidi/Dropbox/python-script/ML/autisme/base/"

# nom de la base 
INPUT_CSV = os.path.join(BASE_PATH,"ASD-ML5.csv")

# importation de la base 
pop = pd.read_csv(INPUT_CSV,sep = ';',decimal=',') 

# definition de la base pop 2 (avec la quality check et sans le petit centre 30) 
pop2 = pop[(pop.site != 30 ) & (pop.sexe == 1) & (pop.CEREB_1OK_2CUT_3DOUTECUT_4FOIRE_5DOUTEQUALI_6KYSTE != 9) ]     

##### essai avec la base de pietro           
pop2 = pd.read_csv('/Users/charleslaidi/Dropbox/python-script/ML/autisme/base/asd_final-pietro-icv.csv', index_col = 'PatientID', sep = ";")
           
           
# =============================================================================
# RÉSIDUALISATION DES FEATURES 
# =============================================================================
          
# création d'une matrice avec le codage binaire des features + age et ICV
cov=patsy.dmatrix('age + ICV + C(sexe)+ C(site)', data=pop2)

# transforme la matrice en array
cov=np.asarray(cov)

# transforme la base de données en array
data=np.asarray(pop2)

# sélectionne la partie de la base avec les variables d'intérêt (variables dépendantes)
freesurfer=data[:,7:150]

##### changement des variables 
#list = pop2.columns.values.tolist()
#list2= list[4:158]
#freesurfer=data[:,4:162]

# fait tourner le modèle linéaire 
pred=linear_model.LinearRegression().fit(cov,freesurfer).predict(cov)

# calcul des résidus comme la différence entre la variable dépendante et la valeur prédite
res=freesurfer-pred

# =============================================================================
# SVM SIMPLE avec K-fold pour diagnostic ASD (code = 1) vs CTRL (code = 4)
# =============================================================================

# -----------------------------------------------------------------------------
# Définition de X et y 
# -----------------------------------------------------------------------------
# X = features avec lesquels on essaye de prédire = les résidus 
# y = variable que l'on cherche à prédire = le diagnostic 

# défintion de X
X=res


# définition de y 
y=data[:,5].astype(int) 

# -----------------------------------------------------------------------------
# Définition des listes qui stockent les valeurs vraies et prédite 
# -----------------------------------------------------------------------------

# accuracy = précision (moyenne entre la sensibilité et la spécificité)
acc=list()

# liste des valeur prédites 
list_predict=list()

# liste des valeur vraies que l'on va comparer avec les valeurs prédites  
list_true=list()

# -----------------------------------------------------------------------------
# Définition du score de balanced acc
# -----------------------------------------------------------------------------

def balanced_acc(t, p):
    recall_scores = recall_score(t,p,pos_label=None, average=None,labels=[1,4])
    ba = recall_scores.mean()
    return ba

# The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.
# The best value is 1 and the worst value is 0.
# moyenne des recall scores 
# en fonction de si on sélectionne 1 ou 4 (ASD ou HS) la fonction permet de calculer la sensibilité ou la spécificité 
# t doit être true et p doit être predict mais je ne suis pas sur 
# je crois que la différence entre balanced_acc et acc c'est que balanced acc prend en compte
# un éventuel déséquilibre entre les groupes 

# -----------------------------------------------------------------------------
# Définition de la fonction de classification (sickt-learn)
# -----------------------------------------------------------------------------

# la fonction class weight permet de donner une pondération en fonction de la fréquence des diag
# même si c'est bien pondéré, ça ne coute rien de laisser le mode auto
# attention par la suite clf va changer (peut être qu'il vaut mieux changer de nom)
clf= svm.LinearSVC(fit_intercept=False,class_weight='auto') 
    
# -----------------------------------------------------------------------------
# Description des features les plus utiles pour la classification (pondération)
# -----------------------------------------------------------------------------
           
# dans la SVM ce n'est pas toujours pertinent (effet sapin de Noël)
# svmlin.coef_ donne les coefficiant pour chaque features, plus il est élevé plus le features
# a été utile. 
 
svmlin = svm.LinearSVC()
svmlin.fit(X, y)
y_pred_svmlin = svmlin.predict(X)
errors = y_pred_svmlin != y
#print("Nb errors=%i, error rate=%.2f" % (errors.sum(), errors.sum() / len(y_pred_svmlin))) 
#print(svmlin.coef_) 

# question : pourquoi est-ce à ce stade, pourquoi ça fonctionne alors que je n'ai même
# pas commencé à prédire (à revoir) ######################################################

# -----------------------------------------------------------------------------
# Sélection du paramètre C - Cross validation interne
# -----------------------------------------------------------------------------

# la paramètre C est la marge qui permet de classer les deux groupes
# à priori on ne connait pas sa valeur, on va essayer de sélectionner la meilleure
# valeur possible

# Ce sont les valeurs de C que l'on va tester - on pourrait en prendre un nombre infini
# mais ça ne change pas forcément beaucoup le résultat
parameters={'C':[10e-6,10e-4,10e-3,10e-2,1,10e2,10e4]} 
           
# Définition du score qui permet de sélectionner le paramètre C
# Ici on prend le paramètre C qui permet de d'avoir la meilleure balanced_acc
# on pourrait choisir autre chose en fonction de ce que l'on cherche 
score=make_scorer(balanced_acc,greater_is_better=True) # tu prends le meilleur C qui maximise le balance acc 

# nouvelle definition de clf
# permet d'intégrer les folds internes, on va diviser chaque fold train en 3 parties (on peut le changer), 
# ensuite on va essayer de classer avec tous les paramètres C définis plus haut et on va utiliser 
# ce paramètres par la suite pour le training et prédire ensuite l'échantillon test
# c'est la cross-validation interne. tout test fait dans le modèle, de telle sorte à ce qu'on 
# ne puisse pas savoir quel paramètre de C est utilisé          
clf = grid_search.GridSearchCV(clf,parameters,cv=3,scoring=score)

# on peut tout à fait changer le nombre de fold ici 

# -----------------------------------------------------------------------------
# Définition des folds externes 
# -----------------------------------------------------------------------------

# ici on prend 10 folds (on divise l'échantillon en 10 parties)
# on prend y (ce que l'on cherche à prédire comme argument)
skf = StratifiedKFold(y,10)
          

# -----------------------------------------------------------------------------
# Prédiction
# -----------------------------------------------------------------------------

# train : pour chaque fold, échantillon que l'on utilise pour le test
# test : pour chaque fold, échantillon que l'on utilise pour faire la prédiction
 
for train, test in skf:

    X_train=X[train,:] # définition des features de l'échantillon train dans chaque fold
    X_test=X[test,:] # définition des features de l'échantillon test dans chaque fold
    y_train=y[train] # définition des diag dans l'échantillon train 
    y_test=y[test] # définition des diag dans l'échantillon test 
    list_true.append(y_test.ravel()) # stock les vraies valeur de y à chaque itération
    X_train = X_train.astype(float) # conversion en float (sinon erreur)
    X_test = X_test.astype(float) # conversion en float (sinon erreur)
    scaler = preprocessing.StandardScaler().fit(X_train) # fonction qui permet de centrer les valeurs des résidus (pas indispensable mais bon) 
    X_train = scaler.transform(X_train) # centre de l'échantillon train
    X_test = scaler.transform(X_test) # centre de l'échantillon test 
    clf.fit(X_train,y_train) # apprentissage dans l'échantillon train
    y_pred = clf.predict(X_test) # prédiction des y en utilisant le classifieur 
    list_predict.append(y_pred) # le stocke dans une liste du statut predit pour chqaque sujet 
           
t=np.concatenate(list_true) # construit la liste des valeurs vraies pour chaque fold 
p=np.concatenate(list_predict) # construit la liste des valeurs prédites pour chaque fold 
recall_scores = recall_score(t,p,pos_label=None, average=None,labels=[1,4]) #permet de calculer la précision chez les patients et les témoins (c'est la précision)
acc=metrics.accuracy_score(t,p) # précision (mais ce n'est pas très clair)
balanced_acc= recall_scores.mean() # moyenne entre sensibilité et spécificité 
print(acc)
print(balanced_acc)
