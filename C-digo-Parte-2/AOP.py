import urllib
import pandas as pd
import re, time
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer, word_tokenize
from nltk.stem.porter import PorterStemmer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import random
import matplotlib.pyplot as plt

#Se leen archivos de entrada
ftr = open("polarity.dev", "r")
fts = open("polarity.train", "r")

#Se crea dataframe para datos de entrenamiento
rows = [line.split(" ",1) for line in ftr.readlines()]
train_df = pd.DataFrame(rows, columns=['Sentiment', 'Text'])
train_df['Sentiment'] = train_df['Sentiment'].convert_objects(convert_numeric=True)

#Se crea dataframe para datos de prueba
rows = [line.split(" ",1) for line in fts.readlines()]
test_df = pd.DataFrame(rows, columns=['Sentiment', 'Text'])
test_df['Sentiment'] = test_df['Sentiment'].convert_objects(convert_numeric=True)
 
#Cantidad de registros por cada set de datos
num_train = train_df.shape[0]
num_test = test_df.shape[0]

#Se crea funcion word_extractor para obtener tokens de un texto (con y sin stemming)
#Notar que la funcion recibe como parametros el texto a analizar, junto con la opcion (True) o no (False)
#de realizar stemming 
def word_extractor(text, stemming):
	#Se utiliza algoritmo de Porter para stemming
	stemmer = PorterStemmer()
	#Se obtienen stopwords del idioma ingles
	commonwords = stopwords.words('english')
	text = re.sub(r'([a-z])\1+', r'\1\1', text)
	words = ""

	if stemming:
		#Se realiza lower-casing y stemming
		wordtokens = [stemmer.stem(word.lower()) \
		             for word in word_tokenize(text.decode('utf-8', 'ignore'))]
	else:
		#Se realiza lower-casing, pero no stemming
		wordtokens = [word.lower() for word in word_tokenize(text.decode('utf-8', 'ignore'))]

	#Se eliminan tokens pertenecientes al conjunto de stopwords
	for word in wordtokens:
		if word not in commonwords:
			words += " " + word
	
	return words

#Se crea funcion word_extractor2 para obtener tokens de un texto (usando lematizacion)
def word_extractor2(text, sw):
	wordlemmatizer = WordNetLemmatizer()
	#Se obtienen stopwords del idioma ingles
	commonwords = stopwords.words('english')
	text = re.sub(r'([a-z])\1+', r'\1\1', text)
	words = ""
	#Se realiza lower-casing y lematizacion
	wordtokens = [wordlemmatizer.lemmatize(word.lower()) \
		     for word in word_tokenize(text.decode('utf-8', 'ignore'))]
	
	#Se eliminan tokens pertenecientes al conjunto de stopwords, en caso de que sw == True
	if sw == True:
		for word in wordtokens:
			if word not in commonwords:
				words += " " + word
	else:
		for word in wordtokens:
			words += " " + word	

	return words

#Se obtiene vocabulario de set de entrenamiento
texts_train1 = [word_extractor2(text, True) for text in train_df.Text]
texts_train2 = [word_extractor2(text, False) for text in train_df.Text]
texts_train3 = [word_extractor(text, True) for text in train_df.Text]
#Se obtiene vocabulario de set de prueba
texts_test1 = [word_extractor2(text, True) for text in test_df.Text]
texts_test2 = [word_extractor2(text, False) for text in test_df.Text]
texts_test3 = [word_extractor(text, True) for text in test_df.Text]
vectorizer1 = CountVectorizer(ngram_range=(1,1), binary=False)
vectorizer2 = CountVectorizer(ngram_range=(1,1), binary=False)
vectorizer3 = CountVectorizer(ngram_range=(1,1), binary=False)

#Se genera representacion vectorial
vectorizer1.fit(np.asarray(texts_train1))
vectorizer2.fit(np.asarray(texts_train2))
vectorizer3.fit(np.asarray(texts_train3))
features_train1 = vectorizer1.transform(texts_train1)
features_train2 = vectorizer2.transform(texts_train2)
features_train3 = vectorizer3.transform(texts_train3)
features_test1 = vectorizer1.transform(texts_test1)
features_test2 = vectorizer2.transform(texts_test2)
features_test3 = vectorizer3.transform(texts_test3)
labels_train = np.asarray((train_df.Sentiment.astype(float)+1)/2.0)
labels_test = np.asarray((test_df.Sentiment.astype(float)+1)/2.0)
vocabulario = vectorizer1.get_feature_names()
dist = list(np.array(features_train1.sum(axis=0)).reshape(-1,))

#Se determina la frecuencia de cada token en el vocabulario
word_freq = zip(vocabulario, dist)
word_freq_ordered = sorted(word_freq, key=lambda tup: tup[1]) 
#for tag, count in word_freq_ordered:
#	print count, tag

#Se construye funcion score_model, la cual evalua el desempeno de un determinado clasificador
def score_model(model, x, y, xt, yt, text):
	acc_train = model.score(x,y)
	acc_test = model.score(xt[:-1], yt[:-1])
	print 'Precision datos de entrenamiento %s: %f'%(text, acc_train)
	print 'Precision datos de prueba %s: %f'%(text, acc_test)
	print 'Analisis detallado de resultados sobre set de prueba:'
	print (classification_report(yt, model.predict(xt), target_names = ['+', '-']))

#Implementacion clasificador bayesiano ingenuo binario
def NAIVE_BAYES(x, y, xt, yt):
	model = BernoulliNB()
	model = model.fit(x, y)
	score_model(model, x, y, xt, yt, 'BernoulliNB')
	return model

#Se evalua clasificador bayesiano ingenuo binario sobre datos provistos
#caso 1: filtrando stopwords, con lematizacion
model1 = NAIVE_BAYES(features_train1, labels_train, features_test1, labels_test)
test_pred1 = model1.predict_proba(features_test1)
spl1 = random.sample(xrange(len(test_pred1)), 5)
for text, sentiment in zip(test_df.Text[spl1], test_pred1[spl1]):
	print sentiment, text

#caso 2: sin filtrar stopwords, con lematizacion
model2 = NAIVE_BAYES(features_train2, labels_train, features_test2, labels_test)
estimation1 =  model2.predict(features_test2)

#caso 3: Filtrando stopwords, con stemming
model3 = NAIVE_BAYES(features_train3, labels_train, features_test3, labels_test)

#Implementacion clasificador bayesiano ingenuo multinomial
def MULTINOMIAL(x,y,xt,yt):
	model = MultinomialNB()
	model = model.fit(x,y)
	score_model(model, x, y, xt, yt, "MULTINOMIAL")
	return model

#Se evalua clasificador bayesiano ingenuo multinomial sobre datos provistos
#caso 1: filtrando stopwords, con lematizacion
model1 = MULTINOMIAL(features_train1, labels_train, features_test1, labels_test)
test_pred1 = model1.predict_proba(features_test1)
spl1 = random.sample(xrange(len(test_pred1)), 5)
for text, sentiment in zip(test_df.Text[spl1], test_pred1[spl1]):
	print sentiment, text

#caso 2: sin filtrar stopwords, con lematizacion
model2 = MULTINOMIAL(features_train2, labels_train, features_test2, labels_test)
estimation2 =  model2.predict(features_test2)

#caso 3: Filtrando stopwords, con stemming
model3 = MULTINOMIAL(features_train3, labels_train, features_test3, labels_test)

#Implementacion modelo de regresion logistica regularizado
def LOGIT(x,y,xt,yt, bestvalue):
	start_t = time.time()
	Cs = [0.01, 0.1, 10, 100, 1000]
	if bestvalue == 0:
		for C in Cs:
			print "Usando C= %f"%C
			model = LogisticRegression(penalty='l2', C=C)
			model= model.fit(x,y)
			score_model(model, x, y, xt, yt, "LOGISTIC")
	else:
		model = LogisticRegression(penalty='l2', C=bestvalue)
		model = model.fit(x,y)
		score_model(model,x,y,xt,yt, "LOGISTIC")
		return model
	

#Se evalua modelo para cada valor de C, filtrando stopwords y usando lematizacion
LOGIT(features_train1, labels_train, features_test1, labels_test, 0)

#Se evalua modelo para mejor valor de C (en este caso, C = 10)
model1 = LOGIT(features_train1, labels_train, features_test1, labels_test, 10)
test_pred1 = model1.predict_proba(features_test1)
spl1 = random.sample(xrange(len(test_pred1)), 5)
for text, sentiment in zip(test_df.Text[spl1], test_pred1[spl1]):
	print sentiment, text

#caso 2: sin filtrar stopwords, con lematizacion
model2 = LOGIT(features_train2, labels_train, features_test2, labels_test, 10)
estimation3 =  model2.predict(features_test2)

#caso 3: Filtrando stopwords, con stemming
model3 = LOGIT(features_train3, labels_train, features_test3, labels_test, 10)

#Implementacion de modelo SVM lineal
def SVM(x,y,xt,yt, bestvalue):
	Cs = [0.01, 0.1, 10, 100, 1000]
	if bestvalue == 0:
		for C in Cs:
			print "El valor de C que se esta probando: %f"%C
			model = SVC(C=C, kernel='linear', probability=True)
			model = model.fit(x,y)
			score_model(model, x, y, xt, yt, "SVM")
	else:
		model = SVC(C=bestvalue, kernel='linear', probability=True)
		model = model.fit(x,y)
		score_model(model, x, y, xt, yt, "SVM")
		return model

#Se evalua modelo para cada valor de C, filtrando stopwords y usando lematizacion
SVM(features_train1, labels_train, features_test1, labels_test, 0)

#Se evalua modelo para mejor valor de C (en este caso, C = 0.1)
model1 = SVM(features_train1, labels_train, features_test1, labels_test, 0.1)
test_pred1 = model1.predict_proba(features_test1)
spl1 = random.sample(xrange(len(test_pred1)), 5)
for text, sentiment in zip(test_df.Text[spl1], test_pred1[spl1]):
	print sentiment, text

#caso 2: sin filtrar stopwords, con lematizacion
model2 = SVM(features_train2, labels_train, features_test2, labels_test, 0.1)

#caso 3: Filtrando stopwords, con stemming
model3 = SVM(features_train3, labels_train, features_test3, labels_test, 0.1)
estimation4 =  model2.predict(features_test2)

#Finalmente, se construyen graficos comparativos
#Primero, se obtiene matriz de confusion para cada metodo
cm_bernoulli = confusion_matrix(labels_test, estimation1)
cm_multinomial = confusion_matrix(labels_test, estimation2)
cm_logistic = confusion_matrix(labels_test, estimation3)
cm_svm = confusion_matrix(labels_test, estimation4)

#Se obtiene valores de TP, FP, FN, TN para cada metodo
#Para Bernoulli
TP1 = cm_bernoulli[0][0]
FP1 = cm_bernoulli[0][1]
FN1 = cm_bernoulli[1][0]
TN1 = cm_bernoulli[1][1]

#Para multinomial
TP2 = cm_multinomial[0][0]
FP2 = cm_multinomial[0][1]
FN2 = cm_multinomial[1][0]
TN2 = cm_multinomial[1][1]

#Para logistic regression
TP3 = cm_logistic[0][0]
FP3 = cm_logistic[0][1]
FN3 = cm_logistic[1][0]
TN3 = cm_logistic[1][1]

#Para SVM
TP4 = cm_svm[0][0]
FP4 = cm_svm[0][1]
FN4 = cm_svm[1][0]
TN4 = cm_svm[1][1]

#Se procede a construir grafico
N = 4
TPs = (TP1, TP2, TP3, TP4)
FPs = (FP1, FP2, FP3, FP4)
FNs = (FN1, FN2, FN3, FN4)
TNs = (TN1, TN2, TN3, TN4)

ind = np.arange(N)
width = 1./(1+N)

fig, ax = plt.subplots()
rects1 = ax.bar(ind, TPs, width, color='r')
rects2 = ax.bar(ind + width, FPs, width, color='y')
rects3 = ax.bar(ind + 2*width, FNs, width, color='b')
rects4 = ax.bar(ind + 3*width, TNs, width, color='k')

ax.set_ylabel('Metricas')
ax.set_title('Metricas por cada metodo de clasificacion')
ax.set_xticks(ind + width)
ax.set_xticklabels(('Bayesiano Ingenuo', 'Multinomial', 'Regresion logistica', 'SVM lineal'))

ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]), ('TP', 'FP', 'FN', 'TN'))

plt.show()
