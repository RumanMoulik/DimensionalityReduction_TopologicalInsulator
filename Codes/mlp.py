import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
import pickle
import sys

iters = 2499
lr = 0.000995012968825878
alpha =  0.0001164694418284876
hl_size=21
hl_depth=9
hl=np.full(hl_depth,hl_size)

print("Iterations: ",iters)
print("Learning Rate: ",lr)
print("Alpha: ",alpha)
print("HL Depth: ",hl_depth)
print("HL Size: ",hl_size)
print("Random Seed: ",sys.argv[1])
with open("pca_20.dat") as f:
    data = [line.strip().split() for line in f]


print("MLP with pca data") 
X=np.array([data[i][1:5] for i in range(len(data))],dtype=float)
Y = np.array([int(data[i][0]) for i in range(len(data))],dtype=int).reshape([len(data)])

#clf = MLPClassifier(max_iter=iters, learning_rate_init=lr, alpha=alpha, hidden_layer_sizes=hl, solver='lbfgs')
clf = MLPClassifier(max_iter=iters, learning_rate_init=lr, alpha=alpha, hidden_layer_sizes=hl, random_state=int(sys.argv[1]))
cvf = cross_validate(clf,X,Y,cv=5,return_estimator=True, n_jobs=-1)
#print(round(clf.score(X_test,Y_test),5),end=' ')
print(cvf['fit_time'])
print(cvf['test_score'])

print("average value Summary")
Average = sum(cvf['test_score'])/len(cvf['test_score'])
print(Average)

print("maximum value Summary")
max_ind = np.argmax(cvf['test_score'])
print(max_ind, max(cvf['test_score']))

clf_best = cvf['estimator'][max_ind]
print("score on whole dataset: ",clf_best.score(X,Y))

pickle.dump(clf_best, open("trained_mlp.pkl", 'wb'))