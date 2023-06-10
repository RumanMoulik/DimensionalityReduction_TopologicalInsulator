import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_validate


trees = 2500
lr = 0.02

print("n_estimators=",trees,", learning_rate=",lr)
with open("pca_20.dat") as f:
    data = [line.strip().split() for line in f]
    
dim = len(data[0])-1
    
av=np.zeros([dim])
maxv=np.zeros([dim])

print("AdaBoosting with pca data") 
X=np.array([data[i][1:] for i in range(len(data))])
Y = np.array([int(data[i][0]) for i in range(len(data))]).reshape([len(data)])

for i in range(1,dim+1):
    clf = AdaBoostClassifier(n_estimators=trees, learning_rate=lr)
    cvf = cross_validate(clf,X[:,:i],Y,cv=5)
    #print(round(clf.score(X_test,Y_test),5),end=' ')
    print(cvf)
    av[i-1]=sum(cvf['test_score'])/len(cvf['test_score'])
    maxv[i-1]=max(cvf['test_score'])

with open("data.dat") as f:
    data = [line.strip().split() for line in f]


print("AdaBoosting with raw data")    
X = np.array([data[i][1:-3] for i in range(1,len(data))])
Y = np.array([int(data[i][-1]) for i in range(1,len(data))]).reshape([len(data)-1])
clf=AdaBoostClassifier(n_estimators=trees, learning_rate=lr)
cvf = cross_validate(clf,X,Y,cv=5)
print(cvf)

print("average value Summary")
print("PCA data",av)
Average = sum(cvf['test_score'])/len(cvf['test_score'])
print("raw_data",Average)

print("maximum value Summary")
print("PCA data",maxv)
print("raw_data",max(cvf['test_score']))

with open("adaboost_"+str(trees)+"_"+str(int(lr*100))+".dat","w") as f:
    line = str(max(cvf['test_score']))
    for i in maxv:
        line+=" "+str(i)
    line+="\n"
    f.write(line)
