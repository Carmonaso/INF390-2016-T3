#!/usr/bin/env python
# -*- coding: utf-8 -*-
import urllib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.neighbors import KNeighborsClassifier


####################################################
########## Parte a: Extracción de datos ############
####################################################
train_data_url = "http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/vowel.train"
test_data_url = "http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/vowel.test"
train_data_f = urllib.urlretrieve(train_data_url, "train_data.csv")
test_data_f = urllib.urlretrieve(test_data_url, "test_data.csv")
train_df = pd.DataFrame.from_csv('train_data.csv',header=0,index_col=0)
test_df = pd.DataFrame.from_csv('test_data.csv',header=0,index_col=0)
train_df.head()
test_df.tail()

#Registros en dataframes:
print "Registros conjunto de entrenamiento: ", len(train_df.index)
print "Registros conjunto de testint: ", len(test_df.index)

len_training_set=len(train_df.index)

# train_df[train_df["y"]==1].count()




####################################################
########## Parte b: Preparación de datos ###########
####################################################

X = train_df.ix[:,'x.1':'x.10'].values
y = train_df.ix[:,'y'].values
X_std = StandardScaler().fit_transform(X)




####################################################
########## Parte c: PCA ############################
####################################################

sklearn_pca = PCA(n_components=2)
Xred_pca = sklearn_pca.fit_transform(X_std)
cmap = plt.cm.get_cmap('Set1')
mclasses=(1,2,3,4,5,6,7,8,9)
mcolors = [cmap(i) for i in np.linspace(0,1,10)]
plt.figure(figsize=(12, 8))
for lab, col in zip(mclasses,mcolors):
    plt.scatter(Xred_pca[y==lab, 0],Xred_pca[y==lab, 1],label=lab,c=col)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
leg = plt.legend(loc='upper right', fancybox=True)
plt.show()



####################################################
########## Parte d: LDA ############################
####################################################



sklearn_lda = LDA(n_components=2)
Xred_lda = sklearn_lda.fit_transform(X_std,y)
cmap = plt.cm.get_cmap('Set1')
mclasses=(1,2,3,4,5,6,7,8,9)
mcolors = [cmap(i) for i in np.linspace(0,1,10)]
plt.figure(figsize=(12, 8))
for lab, col in zip(mclasses,mcolors):
    plt.scatter(Xred_lda[y==lab, 0],Xred_lda[y==lab, 1],label=lab,c=col)

plt.xlabel('LDA/Fisher Direction 1')
plt.ylabel('LDA/Fisher Direction 2')
leg = plt.legend(loc='upper right', fancybox=True)
plt.show()

#
# ####################################################
# ########## Parte f: Construcción Clasificador ######
# ####################################################
#
#
# ##############################################################
# ########## Parte g: Comparación desempeño LDA, QDA, K-NN #####
# ##############################################################


#X_std : X entrenamiento
#Y_std : Y entrenamiento

Xtest = test_df.ix[:,'x.1':'x.10'].values
ytest = test_df.ix[:,'y'].values
X_std_test = StandardScaler().fit_transform(Xtest)


############ LDA #####################
#Construcción y Fit del modelo LDA
lda_model = LDA()
lda_model.fit(X_std,y)
#Score conjunto de entrenamiento y conjunto de testing.
print lda_model.score(X_std,y)
print lda_model.score(X_std_test,ytest)


############ QDA #####################
#Construcción y Fit del modelo QDA
qda_model = QDA()
qda_model.fit(X_std,y)
#Score conjunto de entrenamiento y conjunto de testing.
print qda_model.score(X_std,y)
print qda_model.score(X_std_test,ytest)

# ############ KNN #####################
# #Construcción y Fit del modelo KNN
# knn_model = KNeighborsClassifier(n_neighbors=10)
# knn_model.fit(X_std,y)
# #Score conjunto de entrenamiento y conjunto de testing.
# print knn_model.score(X_std,y)
# print knn_model.score(X_std_test,ytest)
#
#
# score_training=[]
# score_test=[]
# Lclasses=range(1,len_training_set+1)
# #Comportamiento KNN
# for mclass in Lclasses:
#     knn_model = KNeighborsClassifier(n_neighbors=mclass)
#     knn_model.fit(X_std,y)
#
#     score_training.append(knn_model.score(X_std,y))
#     score_test.append(knn_model.score(X_std_test,ytest))
#
# # plt.axis([0,11,0.45,1.03])
# plt.plot(Lclasses, score_training, label="Training")
# plt.plot(Lclasses, score_test, label="Test")
# plt.legend()
# plt.xlabel('Clases')
# plt.ylabel('Score')
#
# plt.show()






######################################################################
########## Parte h: Comparación desempeño con PCA: LDA, QDA, K-NN ####
######################################################################

Lclasses=range(1,10)
score_training_QDA=[]
score_test_QDA=[]

score_training_LDA=[]
score_test_LDA=[]

score_training_KNN=[]
score_test_KNN=[]

for mclass in Lclasses:
    sklearn_pca = PCA(n_components=mclass)
    X_PCA = sklearn_pca.fit_transform(X_std)
    X_PCA_test = sklearn_pca.fit_transform(X_std_test)


    knn_model = KNeighborsClassifier(n_neighbors=10)
    knn_model.fit(X_PCA,y)
    score_training_KNN.append(knn_model.score(X_PCA,y))
    score_test_KNN.append(knn_model.score(X_PCA_test,ytest))

    qda_model = QDA()
    qda_model.fit(X_PCA,y)
    score_training_QDA.append(qda_model.score(X_PCA,y))
    score_test_QDA.append(qda_model.score(X_PCA_test,ytest))

    lda_model= LDA()
    lda_model.fit(X_PCA,y)
    score_training_LDA.append(lda_model.score(X_PCA,y))
    score_test_LDA.append(lda_model.score(X_PCA_test,ytest))








plt.plot(Lclasses, score_training_LDA, label="Training")
plt.plot(Lclasses, score_test_LDA, label="Test")
plt.legend()
plt.xlabel('Clases')
plt.ylabel('Score')
plt.show()


plt.plot(Lclasses, score_training_QDA, label="Training")
plt.plot(Lclasses, score_test_QDA, label="Test")
plt.legend()
plt.xlabel('Clases')
plt.ylabel('Score')
plt.show()

plt.plot(Lclasses, score_training_KNN, label="Training")
plt.plot(Lclasses, score_test_KNN, label="Test")
plt.legend()
plt.xlabel('Clases')
plt.ylabel('Score')
plt.show()









# ######################################################################
# ########## Parte i: Comparación desempeño con LDA: LDA, QDA, K-NN ####
# ######################################################################


dimensiones = len(train_df.columns)
dimensiones=range(1,dimensiones)

score_training_QDA=[]
score_test_QDA=[]

score_training_LDA=[]
score_test_LDA=[]

score_training_KNN=[]
score_test_KNN=[]

for dimension in dimensiones:

    sklearn_lda = LDA(n_components=dimension)
    X_LDA = sklearn_lda.fit_transform(X_std,y)
    X_LDA_test = sklearn_lda.fit_transform(X_std_test,ytest)


    knn_model = KNeighborsClassifier(n_neighbors=10)
    knn_model.fit(X_LDA,y)
    score_training_KNN.append(knn_model.score(X_LDA,y))
    score_test_KNN.append(knn_model.score(X_LDA_test,ytest))

    qda_model = QDA()
    qda_model.fit(X_LDA,y)
    score_training_QDA.append(qda_model.score(X_LDA,y))
    score_test_QDA.append(qda_model.score(X_LDA_test,ytest))

    lda_model= LDA()
    lda_model.fit(X_LDA,y)
    score_training_LDA.append(lda_model.score(X_LDA,y))
    score_test_LDA.append(lda_model.score(X_LDA_test,ytest))

    print lda_model.score(X_LDA_test,ytest)





plt.plot(dimensiones, score_training_LDA, label="Training")
plt.plot(dimensiones, score_test_LDA, label="Test")
plt.legend()
plt.xlabel('Clases')
plt.ylabel('Score')
plt.show()


plt.plot(dimensiones, score_training_QDA, label="Training")
plt.plot(dimensiones, score_test_QDA, label="Test")
plt.legend()
plt.xlabel('Clases')
plt.ylabel('Score')
plt.show()

plt.plot(dimensiones, score_training_KNN, label="Training")
plt.plot(dimensiones, score_test_KNN, label="Test")
plt.legend()
plt.xlabel('Clases')
plt.ylabel('Score')
plt.show()
