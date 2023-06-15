import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt

trees = 2500
lr = 0.02

print("n_estimators=",trees,", learning_rate=",lr)
with open("../pca_20.dat") as f:
    data = [line.strip().split() for line in f]
    
dim = len(data[0])-1
    
time=np.zeros([dim])
maxv=np.zeros([dim])

print("AdaBoosting with pca data") 
X=np.array([data[i][1:] for i in range(len(data))])
Y = np.array([int(data[i][0]) for i in range(len(data))]).reshape([len(data)])

for i in range(1,dim+1):
    print("dimension: ",i)
    for j in range(0,10):
        clf = AdaBoostClassifier(n_estimators=trees, learning_rate=lr)
        cvf = cross_validate(clf,X[:,:i],Y,cv=5,n_jobs=5)
        #print(round(clf.score(X_test,Y_test),5),end=' ')
        print(cvf)
        time[i-1]+=sum(cvf['fit_time'])+sum(cvf['score_time'])
        maxv[i-1]+=max(cvf['test_score'])

raw_time=0
raw_maxv=0
with open("../data.dat") as f:
    data = [line.strip().split() for line in f]

print("AdaBoosting with raw data")    
X = np.array([data[i][1:-3] for i in range(1,len(data))])
Y = np.array([int(data[i][-1]) for i in range(1,len(data))]).reshape([len(data)-1])
for i in range(0,10):
    clf=AdaBoostClassifier(n_estimators=trees, learning_rate=lr)
    cvf = cross_validate(clf,X,Y,cv=5,n_jobs=5)
    print(cvf)
    raw_time+=sum(cvf['fit_time'])+sum(cvf['score_time'])
    raw_maxv+=max(cvf['test_score'])

print("\nMaximum value Summary")
print("PCA data",maxv/10)
print("raw_data",raw_maxv/10)

print("\nTime Summary")
print("PCA data",time/10)
print("raw_data",raw_time/10)

with open("adaboost_"+str(trees)+"_"+str(int(lr*100))+".dat","w") as f:
    f.write("Maximum score summary\n")
    f.write("raw data: "+str(raw_maxv/10)+"\n")
    line=""
    for i in maxv:
        line+=" "+str(i/10)
    line+="\n"
    f.write(line)
    
    f.write("Time summary\n")
    f.write("raw data: "+str(raw_time/10)+"\n")
    line=""
    for i in time:
        line+=" "+str(i/10)
    line+="\n"
    f.write(line)
    
dim_x = [i for i in range(1,dim+1)]   
fig = plt.figure()
plt.plot(dim_x,maxv, marker = '.')
plt.axhline(y = raw_maxv, color = 'red', linestyle = '--')
plt.savefig("max_score.png")
fig = plt.figure()
ax_time.plot(dim_x,time, marker = '.')
ax_time.axhline(y = raw_time, color = 'red', linestyle = '--')
plt.savefig("time.png")