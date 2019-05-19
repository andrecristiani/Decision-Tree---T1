# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, export
from sklearn.model_selection import cross_val_score #novidade

import sklearn.metrics

dados = pd.read_csv('Admission_Predict.csv')
del dados['Serial No.']
dataset = dados.values

Y = dataset[:,len(dataset[0])-1] #only the last column
X = dataset[:,:-1] #all columns except the last one

NY = []
i=0
for line in X:
  X[i][5] = int("{0:.0f}".format(round(line[5],2)))
  i = i+1

for item in Y:
  if item > 0.90:
    NY.append('Alta')
  elif item > 0.75:
    NY.append('Media')
  else:
    NY.append('Baixa')

print(NY)

x_train, x_test, y_train, y_test = train_test_split(X, NY, 
                                                    test_size=0.20, 
                                                    random_state=42) # novidade

print(x_train.shape)
print(x_test.shape)

"""Agora, definimos o algoritmo que será utilizado (AD, nesse caso), mas vamos verificar sua acurácia esperada variando parâmetros."""

tr_acc = []
mln_set = range(2,15)

for max_leaf_nodes in mln_set:
  modelo = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, criterion='entropy')
  scores = cross_val_score(modelo, x_train, y_train, cv=10)
  tr_acc.append(scores.mean())
  
best_mln = mln_set[np.argmax(tr_acc)]
print(best_mln)

te_acc = []

for max_leaf_nodes in mln_set:
  modelo = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, criterion='entropy')
  modelo.fit(x_train, y_train)
  score = modelo.score(x_test, y_test)
  print(max_leaf_nodes)
  print("Score {0}\n".format(score))
  y_pred = modelo.predict(x_test)
  te_acc.append(sklearn.metrics.accuracy_score(y_test, y_pred))

export.export_graphviz(modelo,
                       out_file = 'arvore.dot',
                       feature_names = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research'],
                       class_names = ['Alta', 'Media', 'Baixa'],
                       filled = True,
                       leaves_parallel=True)

import matplotlib.pyplot as plt

plt.plot(mln_set,tr_acc, label='Treino')
plt.plot(mln_set,te_acc, label='Teste')
plt.ylabel('Acurácia')
plt.xlabel('k')
plt.legend()

plt.show()