#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 22:29:32 2017

@author: charleslaidi
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import statsmodels.api as sm
from __future__ import print_function
from collections import Counter
from scipy import stats
import numpy as np
import math
import scipy
from scipy.stats.stats import pearsonr
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


# ============================================================================
# Importation de la base
# ============================================================================

# importation
asdbig = pd.read_csv('/Users/charleslaidi/Dropbox/python-script/ML/autisme/base/asd_final_big.csv', index_col = 'PatientID', sep = ";")

# recodage du sexe et du diagnostic 
asdbig.sexe = asdbig.sexe.map({1:'m', 2:'f'}) # codage du sexe
#asdbig.diag = asdbig.diag.map({1:'asd', 2:'hs'}) # codage du diagnostic

# création des variables pour les variables centrées
lis = ['age_center', 'ICVcm3_volbrain_center','age2','age3']

for h in lis: 
	asdbig.loc[:,h] = pd.Series(np.nan, index=dnm.index) 
 
for x in asdbig.index:
    asdbig.loc[x,'age_center'] = asdbig.loc[x,'age'] - asdbig.age.mean()

for x in asdbig.index:
    asdbig.loc[x,'ICV_center'] = asdbig.loc[x,'ICV'] - asdbig.ICV.mean()    

for x in asdbig.index:
    asdbig.loc[x,'age2'] = asdbig.loc[x,'age'] * asdbig.loc[x,'age']  
    
for x in asdbig.index:
    asdbig.loc[x,'age3'] = asdbig.loc[x,'age'] * asdbig.loc[x,'age'] * asdbig.loc[x,'age'] 


# base_qc light
asdqc1 = asdbig[asdbig.total_qc < 4 ]
asdqc1 = asdqc1[asdqc1.age >= 18]

# base_qc hard
asdqc2 = asdbig[asdbig.total_qc < 3 ]
asdqc2 = asdqc2[asdqc2.age >= 18]
# asdqc2 = asdqc2.loc[asdqc2['site'].isin(centerlist)] quand j'enlève ETH, ça fait tout foirer ! 


# définition de la base avec juste les hommes
asd_men1 = asdqc1[asdqc1.sexe == 'm']
asd_men2 = asdqc2[asdqc2.sexe == 'm']

# définiton de la base avec les 3 plus gros sites
asd_3sites1 = asdqc1.loc[asdqc1['site'].isin(['USM','BNI','CRE'])]
                         
# formule pour avoir les colonnes
liste = asdbig.columns.values.tolist()
# diag = col 3
# label = 4 (Lbankssts_surfavg) à 157 (Raccumb) ou 161 (Cervelet)

centerlist = ['OLI','USM','CRE','IU','NYU','BNI','CAK','MAX']

# liste des features pour la fonction exploratoire

liste = asdbig.columns.values.tolist()
liste_totale = liste[4:158] # c'est la bonne
liste_surf = liste[4:72]
liste_thick = liste[72:144]
liste_vol = liste[144:158]

                         
# ============================================================================
# Fonctions
# ============================================================================
                       
 
# ----------------------------------------------------------------------------
# Fonction exploratoire 
# ----------------------------------------------------------------------------                         
                         

# def de la fonction exploratoire

def lmcharlesWO(list,formule,base,cof,p):
    
# setup    
    d={} # création du dictionnaire pour les outliers
    sigpv={} # création du dictionnaire pour les pvalue
    #p = 0.05 # valeur seuil de significativité (Bonferroni pour 75 tests = 0.000666)

# modèle linéaire    
    for var in list:
        #print("-------------------------------------------------------------------------------")
        #print(var) # affiche le nom de la variable
        #print("-------------------------------------------------------------------------------")
        lm_full = sm.formula.ols(formula="%s ~ %s " % (var,formule), data=base).fit() # modèle linéaire
        anova = sm.stats.anova_lm(lm_full, typ=2)
        #pval = anova.loc['diag','PR(>F)'] # changer la variable à laquelle on s'intéresse
        pval = anova.loc['%s'%(cof),'PR(>F)'] # changer la variable à laquelle on s'intéresse
        #print('pval_brut = %s' % pval) # imprime la p value
        residuals = lm_full.outlier_test() # calcul des résidus
        stud = residuals.student_resid
        d["var" + var] = [] # création d'une entrée pour le dictionnaire, qui est une liste qui pourra contenir les noms des outliers
        sigpv[var] = [] # création d'une entrée pour le dictionnaire, pour stocker les p values
        for y in base.index: # loop sur toute la base pour chercher des outliers
            if (stud[y] > 3) or (stud[y] < -3):# conditions pour les outliers
                d["var" + var].append(y) # va permettre de créer une liste avec le nom des outliers
                
# pas d'ouliers
        if len(d["var" + var]) == 0: # si cette liste est vide = si pas d'outliers
            #print("pas d OL") # impression de "pas d'outliers"
            if pval < p: # si le p est inf à 0.05
                print("-------------------------------------------------------------------------------")
                print(var) # affiche le nom de la variable
                print("-------------------------------------------------------------------------------")
                print('pval_brut = %s' % pval) # imprime la p value
                print("pas d OL") # impression de "pas d'outliers"
                stat, pvalnorm = scipy.stats.kstest(stud,'norm')# test de la normalité des résidus
                if pvalnorm < 0.05: # si ce n'est pas normal
                    print("attention distrib. res non norm.") # impression d'une alerte
                print("norm. des résidus : pval = %s" % (pvalnorm)) # impression de la pvalue du KS
                #print("                                            #### Res. Sig seuil %s"%(p))
                sigpv[var].append(pval) # impression d'une alerte (au dessus) et remplissage du dictionnaire

# outliers                
        if len(d["var" + var]) > 0: # si il y a des outliers
            #print("OL detectés") # impression d'une alerte
            #print("Nb d'OL = %s" % len(d["var" + var])) # imprime le nombre d'OL
            #print("nom des OL = %s" % (d["var" + var])) # imprime le nom des OL
            #print("longueur de la base = %s" % (len(base))) # donne la longeur initiale de la base
            base2 = base.drop(d["var" + var]) # crée une nouvelle base sans les outliers
            #print("longueur de la base WO = %s" % (len(base2))) # longueur de la nouvelle base
            lm_full = sm.formula.ols(formula="%s ~ %s" % (var,formule), data=base2).fit() # refait tourner le modèle
            anova = sm.stats.anova_lm(lm_full, typ=2)
            #pval = anova.loc['diag','PR(>F)'] # à changer si jamais je veux regarder autre chose
            pval = anova.loc['%s'%(cof),'PR(>F)'] # à changer si jamais je veux regarder autre chose
            residuals = lm_full.outlier_test() # résidus
            stud = residuals.student_resid
            #print('pval_wo = %s' % pval) # nouvelle pvalue après avoir enlevé les outliers
            if pval < p:
                print("-------------------------------------------------------------------------------")
                print(var) # affiche le nom de la variable
                print("-------------------------------------------------------------------------------")
                print('pval_wo = %s' % pval) # nouvelle pvalue après avoir enlevé les outliers
                print("%s OLs " % len(d["var" + var])) # imprime le nombre d'OL
                print("nom des OL = %s" % (d["var" + var])) # imprime le nom des OL
                print("longueur de la base = %s" % (len(base))) # donne la longeur initiale de la base
                print("longueur de la base WO = %s" % (len(base2))) # longueur de la nouvelle base
                sigpv[var].append(pval)
                stat, pvalnorm = scipy.stats.kstest(stud,'norm') # test du KS
                
                print("norm. des résidus : pval = %s" % (pvalnorm)) 
                if pvalnorm < 0.05:
                    print("attention distrib. res non norm.")
                    print("norm. des résidus : pval = %s" % (pvalnorm)) # impression de la pvalue du KS

# Résumé des résultats            
    print("===============================================================================")            
    print("Résumé des résultats")
    print("===============================================================================")
    for var in list:
        if len(sigpv[var]) > 0: # si la pvalue est significative
            print("%s = %s" % (var,sigpv[var])) # impression de la variable et de sa pvalue

# lmcharlesWO(list2,'age + Site + age + ICV + diag + age * diag',ds3sites_man,'age:diag',0.05)
 

 
# ----------------------------------------------------------------------------
# ML svm simple
# ----------------------------------------------------------------------------                         
                         
def MLcharles(base,CVext=int, CVint=int, formule=str,code_patient=int,code_temoin=int,colone_diag=int,borne_inf=int,borne_sup=int):
    # résidualisation
    cov=patsy.dmatrix('%s' % (formule), data=base)
    cov=np.asarray(cov)
    data=np.asarray(base)
    freesurfer=data[:,borne_inf:borne_sup]
    pred=linear_model.LinearRegression().fit(cov,freesurfer).predict(cov)
    res=freesurfer-pred
    # Codage des variables
    X=res
    y=data[:,colone_diag].astype(int) 
    # def des listes pour les résultats
    acc=list()
    list_predict=list()
    list_true=list()
    # calcul de la précision
    def balanced_acc(t, p):
        recall_scores = recall_score(t,p,pos_label=None, average=None,labels=[code_patient,code_temoin])
        ba = recall_scores.mean()
        return ba
    # modèle basique de ML
    clf= svm.LinearSVC(fit_intercept=False,class_weight='balanced') 
    # pondération des features     
    svmlin = svm.LinearSVC()
    svmlin.fit(X, y)
    y_pred_svmlin = svmlin.predict(X)
    errors = y_pred_svmlin != y
    #print("Nb errors=%i, error rate=%.2f" % (errors.sum(), errors.sum() / len(y_pred_svmlin))) 
    #print(svmlin.coef_) 
    # définition des paramètres
    parameters={'C':[10e-6,10e-4,10e-3,10e-2,1,10e2,10e4]} 
    score=make_scorer(balanced_acc,greater_is_better=True) # tu prends le meilleur C qui maximise le balance acc 
    clf = grid_search.GridSearchCV(clf,parameters,cv=CVint,scoring=score)
    skf = StratifiedKFold(y,CVext)
    # prediction           
    for train, test in skf:
        X_train=X[train,:] # définition des features de l'échantillon train dans chaque fold
        X_test=X[test,:] # définition des features de l'échantillon test dans chaque fold
        y_train=y[train] # définition des diag dans l'échantillon train 
        y_test=y[test] # définition des diag dans l'échantillon test 
        list_true.append(y_test.ravel()) # stock les vraies valeur de y à chaque itération
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        scaler = preprocessing.StandardScaler().fit(X_train) # fonction qui permet de centrer les valeurs des résidus (pas indispensable mais bon) 
        X_train = scaler.transform(X_train) # centre de l'échantillon train
        X_test = scaler.transform(X_test) # centre de l'échantillon test 
        clf.fit(X_train,y_train) # apprentissage dans l'échantillon train
        y_pred = clf.predict(X_test) # prédiction des y en utilisant le classifieur 
        list_predict.append(y_pred) # le stocke dans une liste du statut predit pour chqaque sujet 
    # calcul du score de prédiction           
    t=np.concatenate(list_true) # construit la liste des valeurs vraies pour chaque fold 
    p=np.concatenate(list_predict) # construit la liste des valeurs prédites pour chaque fold 
    recall_scores = recall_score(t,p,pos_label=None, average=None,labels=[code_patient,code_temoin]) #permet de calculer la précision chez les patients et les témoins (c'est la précision)
    acc=metrics.accuracy_score(t,p) # précision (mais ce n'est pas très clair)
    balanced_acc= recall_scores.mean() # moyenne entre sensibilité et spécificité 
    print("=========================================")
    print("Accuracy = %s" % (acc))
    print("Balanced Accuracy = %s" % (balanced_acc))
    print("=========================================")

# ----------------------------------------------------------------------------
# ML univariate feature selection
# ----------------------------------------------------------------------------                         
    

def MLcharles_FS(base,CVext=int, CVint=int, formule=str,code_patient=int,code_temoin=int,colone_diag=int,borne_inf=int,borne_sup=int):
    # résidualisation
    cov=patsy.dmatrix('%s' % (formule), data=base)
    cov=np.asarray(cov)
    data=np.asarray(base)
    freesurfer=data[:,borne_inf:borne_sup]
    pred=linear_model.LinearRegression().fit(cov,freesurfer).predict(cov)
    res=freesurfer-pred
    # Codage des variables
    X=res
    y=data[:,colone_diag].astype(int) 
    # Calcul du score
    def balanced_acc(t, p):
        recall_scores = recall_score(t,p,pos_label=None, average=None,labels=[code_patient,code_temoin])
        ba = recall_scores.mean()
        return ba
    scores=np.zeros((3,100))
    acc=list()
    for f in range(1,100): # pour les 1% des "meilleurs" features, 
        n=0
        list_predict=list()
        list_true=list()
        #L1 SFVM clf
        #clf= svm.LinearSVC(fit_intercept=False,class_weight='auto',dual=False,penalty='l1') 
        #L2 SVM clf:
        clf= svm.LinearSVC(fit_intercept=False,class_weight='balanced') 
        parameters={'C':[10e-6,10e-4,10e-3,10e-2,1,10e2,10e4]} 
        score=make_scorer(balanced_acc,greater_is_better=True)
        clf = grid_search.GridSearchCV(clf,parameters,cv=CVint,scoring=score)
        skf = StratifiedKFold(y,CVext)
        # début de l'entrainement et de la prédiction
        for train, test in skf:
                X_train=X[train,:]
                X_test=X[test,:]
                y_train=y[train]
                y_test=y[test] 
                list_true.append(y_test.ravel())
                X_train = X_train.astype(float)
                X_test = X_test.astype(float)
                scaler = preprocessing.StandardScaler().fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
            
                selector = SelectPercentile(f_classif, percentile=f) # va ordoner les meilleurs features 
                clf.fit(selector.fit_transform(X_train,y_train.ravel()), y_train.ravel()) # dis moi lesquels sont les plus selectif et renvoie moi juste le x des top features slt sur le train
                X_test=selector.transform(X_test) # transforme mon test 
                y_pred = clf.predict(X_test)
                list_predict.append(y_pred)
            
        t=np.concatenate(list_true)
        p=np.concatenate(list_predict)
        recall_scores = recall_score(t,p,pos_label=None, average=None,labels=[code_patient,code_temoin])
        scores[0,f]=recall_scores.mean()
        scores[1,f]=recall_scores[0]
        scores[2,f]=recall_scores[1]  
        #metrics.roc_auc_score(t,p)
        
        print(f)
                                     
    import matplotlib.pyplot as plt
    plt.plot(scores[0,:],label="Accuracy")
    plt.plot(scores[1,:],label="Specificity")
    plt.plot(scores[2,:],label="Sensitivity")
    plt.xlabel('Percentiles')
    plt.ylabel('Scores')
    plt.legend(loc='lower right')

   
# ----------------------------------------------------------------------------
# ML calcul de la p-value
# ----------------------------------------------------------------------------                         
    
def MLcharles_PV(base,CVext=int, CVint=int, formule=str,code_patient=int,code_temoin=int,colone_diag=int,borne_inf=int,borne_sup=int,nper=int,seuil=float):
    nperms=nper
    scores_perm = list()
    recall_perm = list()
        
    
    for n in xrange(nperms):
        cov=patsy.dmatrix('%s' % (formule), data=base)
        cov=np.asarray(cov)
        data=np.asarray(base)
        freesurfer=data[:,borne_inf:borne_sup]
        pred=linear_model.LinearRegression().fit(cov,freesurfer).predict(cov)
        res=freesurfer-pred
        # Codage des variables
        X=res
        y=data[:,colone_diag].astype(int) 
        y = np.random.permutation(y) # shuffle 1000 fois et donne 1000 fois un taux de prédiction
        # def des listes pour les résultats
        acc=list()
        list_predict=list()
        list_true=list()
        # calcul de la précision
        def balanced_acc(t, p):
            recall_scores = recall_score(t,p,pos_label=None, average=None,labels=[code_patient,code_temoin])
            ba = recall_scores.mean()
            return ba
        # modèle basique de ML
        clf= svm.LinearSVC(fit_intercept=False,class_weight='balanced') 
        # pondération des features     
        svmlin = svm.LinearSVC()
        svmlin.fit(X, y)
        y_pred_svmlin = svmlin.predict(X)
        errors = y_pred_svmlin != y
        #print("Nb errors=%i, error rate=%.2f" % (errors.sum(), errors.sum() / len(y_pred_svmlin))) 
        #print(svmlin.coef_) 
        # définition des paramètres
        parameters={'C':[10e-6,10e-4,10e-3,10e-2,1,10e2,10e4]} 
        score=make_scorer(balanced_acc,greater_is_better=True) # tu prends le meilleur C qui maximise le balance acc 
        clf = grid_search.GridSearchCV(clf,parameters,cv=CVint,scoring=score)
        skf = StratifiedKFold(y,CVext)
        # prediction           
        for train, test in skf:
            X_train=X[train,:] # définition des features de l'échantillon train dans chaque fold
            X_test=X[test,:] # définition des features de l'échantillon test dans chaque fold
            y_train=y[train] # définition des diag dans l'échantillon train 
            y_test=y[test] # définition des diag dans l'échantillon test 
            list_true.append(y_test.ravel()) # stock les vraies valeur de y à chaque itération
            X_train = X_train.astype(float)
            X_test = X_test.astype(float)
            scaler = preprocessing.StandardScaler().fit(X_train) # fonction qui permet de centrer les valeurs des résidus (pas indispensable mais bon) 
            X_train = scaler.transform(X_train) # centre de l'échantillon train
            X_test = scaler.transform(X_test) # centre de l'échantillon test 
            clf.fit(X_train,y_train) # apprentissage dans l'échantillon train
            y_pred = clf.predict(X_test) # prédiction des y en utilisant le classifieur 
            list_predict.append(y_pred) # le stocke dans une liste du statut predit pour chqaque sujet 
        # calcul du score de prédiction           
        t=np.concatenate(list_true) # construit la liste des valeurs vraies pour chaque fold 
        p=np.concatenate(list_predict) # construit la liste des valeurs prédites pour chaque fold 
        recall_scores = recall_score(t,p,pos_label=None, average=None,labels=[code_patient,code_temoin]) #permet de calculer la précision chez les patients et les témoins (c'est la précision)
        acc=metrics.accuracy_score(t,p) # précision (mais ce n'est pas très clair)
        balanced_acc= recall_scores.mean() # moyenne entre sensibilité et spécificité 
        scores_perm.append(acc)
        recall_perm.append(recall_score(t,p,pos_label=None, average=None,labels=[code_patient,code_temoin]))
        print(acc)     
        
    scores_perm=np.array(scores_perm)
    pval=np.sum(scores_perm >=0.23)/float(nperms)
    print(pval)
    
    recall_perm=np.array(recall_perm)
    spe=recall_perm[:,0]
    sen=recall_perm[:,1]
    pval=np.sum(spe >=seuil)/float(nperms)
    print(pval)
    pval=np.sum(sen >=seuil)/float(nperms)
    print(pval)
    
    
    plt.hist(scores_perm, 10, label='Permutation scores')
    plt.plot(2 * [seuil],plt.ylim(),'--g', linewidth=3)
    plt.xlabel('Accuracy')
    
#    
#    plt.hist(sen, 10, label='Permutation scores')
#    plt.plot(2 * [seuil],plt.ylim(),'--g', linewidth=3)
#    plt.xlabel('Specificity')
#    
#    
#    plt.hist(sen, 10, label='Permutation scores')
#    plt.plot(2 * [seuil],plt.ylim(),'--g', linewidth=3)
#    plt.xlabel('Sensitivity')
    

    
# -----------------------------------------------------------------------------
# Modèle linéaire simple + résidus + partial residual plot
# -----------------------------------------------------------------------------

def lmcharles(var,formule,base):
    lm_full = sm.formula.ols(formula="%s ~ %s" % (var,formule), data=base).fit()
    print("===============================================================================")
    print(" formule = %s ~ %s" % (var,formule))
    print("===============================================================================")
    print(lm_full.summary())  
    residuals = lm_full.outlier_test() 
    stud = residuals.student_resid
    print("-------------------------------------------------------------------------------")
    print('test normalité python')
    print(scipy.stats.mstats.normaltest(stud))
    print('KS test python')
    print(scipy.stats.kstest(stud,'norm'))
    print("-------------------------------------------------------------------------------")
    plt.hist(stud)
    fig, ax = plt.subplots(figsize=(3, 3))
    fig = sm.graphics.plot_ccpr(lm_full, 'diag[T.hs]', ax=ax)
           
              
# ============================================================================
# Machine learning modèle SVM simple 
# ============================================================================

# -----------------------------------------------------------------------------
# Tous features avec régression sur tous les features sur la base entière
# -----------------------------------------------------------------------------
                         
# Machine-Learning ; ACC = 61% 
MLcharles(asdqc2,CVext=10,CVint=3,formule=('age + C(site) + C(sexe) + ICV'),code_patient=1,code_temoin=2,colone_diag=3,borne_inf=4,borne_sup=157)

# Base homme ; ACC = 61%
MLcharles(asd_men2,CVext=10,CVint=3,formule=('age + C(site) + ICV'),code_patient=1,code_temoin=2,colone_diag=3,borne_inf=4,borne_sup=157)

# Features = CT sans régression sur l'ICV 
# base totale ; ACC = 61%
MLcharles(asdqc2,CVext=10,CVint=3,formule=('age + C(site) + C(sexe)'),code_patient=1,code_temoin=2,colone_diag=3,borne_inf=72,borne_sup=142)
# base homme ; ACC = 58%
MLcharles(asd_men2,CVext=10,CVint=3,formule=('age + C(site)'),code_patient=1,code_temoin=2,colone_diag=3,borne_inf=72,borne_sup=142)
# pas de changement quand on enlève l'ICV

# Features = SurfAvg avec régression sur l'ICV (sans surf moy)
# base totale ; ACC = 52%
MLcharles(asdqc2,CVext=10,CVint=3,formule=('age + C(site) + C(sexe) + ICV'),code_patient=1,code_temoin=2,colone_diag=3,borne_inf=4,borne_sup=71)
# base homme ; ACC = 52%
MLcharles(asd_men2,CVext=10,CVint=3,formule=('age + C(site) + ICV'),code_patient=1,code_temoin=2,colone_diag=3,borne_inf=4,borne_sup=71)

# Features = Volume avec régression sur l'ICV
# base totale ; ACC = 50%
MLcharles(asdqc2,CVext=10,CVint=3,formule=('age + C(site) + C(sexe) + ICV'),code_patient=1,code_temoin=2,colone_diag=3,borne_inf=144,borne_sup=155)
# base homme ; ACC = 52%
MLcharles(asd_men2,CVext=10,CVint=3,formule=('age + C(site) + ICV'),code_patient=1,code_temoin=2,colone_diag=3,borne_inf=144,borne_sup=155)


# Tous features avec régression sur tous les features sur la base entière

# Machine-Learning ; ACC = 61% 
MLcharles(asdqc2,CVext=10,CVint=3,formule=('age + C(site) + C(sexe) + ICV'),code_patient=1,code_temoin=2,colone_diag=3,borne_inf=4,borne_sup=157)

# Base homme ; ACC = 61%
MLcharles(asd_men2,CVext=10,CVint=3,formule=('age + C(site) + ICV'),code_patient=1,code_temoin=2,colone_diag=3,borne_inf=4,borne_sup=157)

# Features = CT sans régression sur l'ICV 
# base totale ; ACC = 61%
MLcharles(asdqc2,CVext=10,CVint=3,formule=('age + C(site) + C(sexe)'),code_patient=1,code_temoin=2,colone_diag=3,borne_inf=72,borne_sup=142)
# base homme ; ACC = 58%
MLcharles(asd_men2,CVext=10,CVint=3,formule=('age + C(site)'),code_patient=1,code_temoin=2,colone_diag=3,borne_inf=72,borne_sup=142)
# pas de changement quand on enlève l'ICV

# Features = SurfAvg avec régression sur l'ICV (sans surf moy)
# base totale ; ACC = 52%
MLcharles(asdqc2,CVext=10,CVint=3,formule=('age + C(site) + C(sexe) + ICV'),code_patient=1,code_temoin=2,colone_diag=3,borne_inf=4,borne_sup=71)
# base homme ; ACC = 52%
MLcharles(asd_men2,CVext=10,CVint=3,formule=('age + C(site) + ICV'),code_patient=1,code_temoin=2,colone_diag=3,borne_inf=4,borne_sup=71)

# Features = Volume avec régression sur l'ICV
# base totale ; ACC = 50%
MLcharles(asdqc2,CVext=10,CVint=3,formule=('age + C(site) + C(sexe) + ICV'),code_patient=1,code_temoin=2,colone_diag=3,borne_inf=144,borne_sup=155)
# base homme ; ACC = 52%
MLcharles(asd_men2,CVext=10,CVint=3,formule=('age + C(site) + ICV'),code_patient=1,code_temoin=2,colone_diag=3,borne_inf=144,borne_sup=155)



# Site par site avec hommes
# CVext = 5, CV int = 5                
OLIms = asd_men2.loc[asd_men2['site'].isin(['OLI'])] # 14 ASD, 15 HS - 51%
USMms = asd_men2.loc[asd_men2['site'].isin(['USM'])] # 39 ASD, 29 HS - 58%            
CREms = asd_men2.loc[asd_men2['site'].isin(['CRE'])] # 28 ASD, 19 HS - 58%
IUms = asd_men2.loc[asd_men2['site'].isin(['IU'])]   # 11 ASD, 13 HS - 41%
NYUms = asd_men2.loc[asd_men2['site'].isin(['NYU'])] # 13 ASD, 24 HS - 54%
BNIms = asd_men2.loc[asd_men2['site'].isin(['BNI'])] # 28 ASD, 27 HS - 56%
CALms = asd_men2.loc[asd_men2['site'].isin(['CAL'])] # 10 ASD, 12 HS - 55%
ETHms = asd_men2.loc[asd_men2['site'].isin(['ETH'])] # 9 ASD, 21 HS (à enlever ++) - 40%
MAXms = asd_men2.loc[asd_men2['site'].isin(['MAX'])] # 11 ASD, 19 HS - 75% 

# Site par site avec hommes + femmes
# CVext = 5, CV int = 5                
OLIs = asdqc2.loc[asdqc2['site'].isin(['OLI'])] # 14 ASD, 15 HS - 51%
USMs = asdqc2.loc[asdqc2['site'].isin(['USM'])] # 39 ASD, 29 HS - 58%            
CREs = asdqc2.loc[asdqc2['site'].isin(['CRE'])] # 28 ASD, 19 HS - 58%
IUs = asdqc2.loc[asdqc2['site'].isin(['IU'])]   # 11 ASD, 13 HS - 41%
NYUs = asdqc2.loc[asdqc2['site'].isin(['NYU'])] # 13 ASD, 24 HS - 54%
BNIs = asdqc2.loc[asdqc2['site'].isin(['BNI'])] # 28 ASD, 27 HS - 56%
CALs = asdqc2.loc[asdqc2['site'].isin(['CAL'])] # 10 ASD, 12 HS - 55%
ETHs = asdqc2.loc[asdqc2['site'].isin(['ETH'])] # 9 ASD, 21 HS (à enlever ++) - 40%
MAXs = asdqc2.loc[asdqc2['site'].isin(['MAX'])] # 11 ASD, 19 HS - 75%                

# liste des centres 
list_site_hommes = [OLIms,USMms,CREms,IUms,NYUms,BNIms,CALms,ETHms,MAXms]                  
list_site = [OLIs,USMs,CREs,IUs,NYUs,BNIs,CALs,ETHs,MAXs]  

# ML centre par centre
for x in list_site_hommes:
    print('Centre = %s' %(Counter(x.site)))
    MLcharles(x,CVext=5,CVint=5,formule=('ICV + age + sexe'),code_patient=1,code_temoin=2,colone_diag=3,borne_inf=4,borne_sup=157)

MLcharles_PV(MAXms,CVext=5,CVint=3,formule=('age + ICV'),code_patient=1,code_temoin=2,colone_diag=3,borne_inf=4,borne_sup=157,nper=100,seuil=0.75)
   
    
# =============================================================================    
# ML meilleurs centres
# =============================================================================

# -----------------------------------
# 3 meilleurs centres hommes - 68% !!
# -----------------------------------
best2 = asd_men2.loc[asd_men2['site'].isin(['MAX','USM','CRE'])]
# SVM                     
MLcharles(best2,CVext=10,CVint=5,formule=('age + C(site) + ICV'),code_patient=1,code_temoin=2,colone_diag=3,borne_inf=4,borne_sup=157)
# feature selection
MLcharles_FS(best2,CVext=15,CVint=5,formule=('age + C(site) + ICV'),code_patient=1,code_temoin=2,colone_diag=3,borne_inf=4,borne_sup=157)
# p value shuffle
MLcharles_PV(best2,CVext=10,CVint=5,formule=('age + C(site) + ICV'),code_patient=1,code_temoin=2,colone_diag=3,borne_inf=4,borne_sup=157,nper=100,seuil=0.68)

# Essai avec CT 57%
MLcharles(best2,CVext=10,CVint=5,formule=('age + C(site)'),code_patient=1,code_temoin=2,colone_diag=3,borne_inf=72,borne_sup=142)

# 3 meilleurs centres hommes/femmes - 63%
best2 = asdqc2.loc[asdqc2['site'].isin(['MAX','USM','CRE'])]    
MLcharles(best2,CVext=10,CVint=5,formule=('age + ICV + C(site) + C(sexe)'),code_patient=1,code_temoin=2,colone_diag=3,borne_inf=4,borne_sup=157)

# ne marche pas mieux avec asdqc1
# marche mieux avec les hommes seuls > hommes + femmes

best2 = asd_men1.loc[asd_men1['site'].isin(['MAX','USM','CRE'])]    
MLcharles(best2,CVext=10,CVint=5,formule=('age + C(site) + ICV'),code_patient=1,code_temoin=2,colone_diag=3,borne_inf=72,borne_sup=147)

# idées : est ce que c'est dans les groupes ou il y a le plus de variance chez les asd qu'on 
# essayer avec la base non qc (asdqc1 = c'est moins bien)
# essayer avec les hommes et les femmes = c'est moins bien
# essayer avec juste CT sans ICV ou avec ICV = c'est moins bien

# essayer en combinant toutes les possibilités
# essayer en faisant de la univariate feature selection
# essayer avec freesurfer 6 et tous les types de mesure possibles et imaginable = ? 


# ccl : supérieur à la chance


# =============================================================================
# Régression linéaire
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Exploration de la base
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# A. Surf avg = 34 tests de chaque côté 
# -----------------------------------------------------------------------------
# diag
lmcharlesWO(liste_surf,'age + site + sexe + ICV + diag',asdqc2,'diag',0.001)
lmcharlesWO(liste_surf,'age + site + ICV + diag',asd_men2,'diag',0.001)

# age
lmcharlesWO(liste_surf,'diag * age2 + site + sexe + ICV',asdqc2,'diag:age2',0.05)

# -----------------------------------------------------------------------------
# B. Cor thick = 36 tests de chaque côté
# -----------------------------------------------------------------------------
# diag
lmcharlesWO(liste_thick,'age + site + sexe + diag',asdqc2,'diag',0.05)
# age
lmcharlesWO(liste_thick,'site + sexe + diag * age',asdqc2,'diag:age',0.05)

# -----------------------------------------------------------------------------
# C. Volumes = 7 tests de chaque côté 
# -----------------------------------------------------------------------------
# diag
lmcharlesWO(liste_vol,'age + site + sexe + diag + ICV',asdqc2,'diag',0.05)
# age
lmcharlesWO(liste_vol,'site + sexe + diag * age + ICV',asdqc2,'diag:age',0.05)

# -----------------------------------------------------------------------------
# 2. Exploration des résultats
# -----------------------------------------------------------------------------

from sklearn import cluster

>>> iris = datasets.load_iris()
>>> X_iris = iris.data
>>> y_iris = iris.target

>>> k_means = cluster.KMeans(n_clusters=3)
>>> k_means.fit(X_iris) 
KMeans(algorithm='auto', copy_x=True, init='k-means++', ...
>>> print(k_means.labels_[::10])
[1 1 1 1 1 0 0 0 0 0 2 2 2 2 2]
>>> print(y_iris[::10])
[0 0 0 0 0 1 1 1 1 1 2 2 2 2 2]



def ClusCharles(base,CVext=int, CVint=int, formule=str,code_patient=int,code_temoin=int,colone_diag=int,borne_inf=int,borne_sup=int):
    # résidualisation
    cov=patsy.dmatrix('%s' % (formule), data=base)
    cov=np.asarray(cov)
    data=np.asarray(base)
    freesurfer=data[:,borne_inf:borne_sup]
    pred=linear_model.LinearRegression().fit(cov,freesurfer).predict(cov)
    res=freesurfer-pred
    # Codage des variables
    X=res
    
   































                                      
                
             
  
# 3 meilleurs centres - 64%               
best3 = pop3_1.loc[pop3_1['site'].isin(['MAX','USM','CRE'])]    
MLcharles(best3,CVext=7,CVint=3,formule=('age + ICV + C(site)'),code_patient=1,code_temoin=2,colone_diag=3,borne_inf=4,borne_sup=157)

       
# Base homme => 61% (CVext = 10, CVint = 5, age + site + ICV) / 62% (CVext = 20, CVint = 5, age + site + ICV)
pop3_1 = pop3[pop3.sexe == 1]   
MLcharles(pop3_1,CVext=20,CVint=5,formule=('age + C(site)'),code_patient=1,code_temoin=2,colone_diag=3,borne_inf=4,borne_sup=157)
                         
MLcharles(pop3,CVext=10,CVint=3,formule=('age + C(site) + C(sexe) + ICV'),code_patient=1,code_temoin=2,colone_diag=3,borne_inf=4,borne_sup=157)
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
