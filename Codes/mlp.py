import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
import pickle
import sys
from dotenv import dotenv_values
import os
import random

params=dotenv_values("params")
dim = int(params['dim'])
iters = int(params['iters'])
lr = float(params['lr'])
alpha = float(params['alpha'])
hl_size = int(params['hl_size'])
hl_depth = int(params['hl_depth'])
hl=np.full(hl_depth,hl_size)
seeds = random.sample(range(0,100000), 1000)
best_score = 0
with open("pca_20.dat") as f:
    data = [line.strip().split() for line in f]

for i in seeds:
    print("New Model Start------------------------------------------")
    print("Iterations: ",iters)
    print("Learning Rate: ",lr)
    print("Alpha: ",alpha)
    print("HL Depth: ",hl_depth)
    print("HL Size: ",hl_size)
    print("Random Seed: ",i)

    print("MLP with pca data") 
    X=np.array([data[i][1:dim+1] for i in range(len(data))],dtype=float)
    Y = np.array([int(data[i][0]) for i in range(len(data))],dtype=int).reshape([len(data)])

    clf = MLPClassifier(max_iter=iters, learning_rate_init=lr, alpha=alpha, hidden_layer_sizes=hl, random_state=i)
    cvf = cross_validate(clf,X,Y,cv=5,return_estimator=True, n_jobs=5)
    print(cvf['fit_time'])
    print(cvf['test_score'])

    print("maximum value Summary")
    max_ind = np.argmax(cvf['test_score'])
    print(max_ind, max(cvf['test_score']))

    clf_best = cvf['estimator'][max_ind]
    score = clf_best.score(X,Y)
    print("score on whole dataset: ",score)

    if score > best_score:
        best_score=score
        print("Saving this model")
        pickle.dump(clf_best, open("trained_mlp.pkl", 'wb'))
        
    print("-----------------------------------\n")
    
print("Highest score attained: ", best_score)
