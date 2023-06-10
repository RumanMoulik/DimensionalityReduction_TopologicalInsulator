import numpy as np
from sklearn.decomposition import KernelPCA, PCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt

with open("data.dat") as f:
    data = [line.strip().split() for line in f]
    
X=np.array([data[i][1:-3] for i in range(1,len(data))])
#print(np.shape(X))
samples, features = np.shape(X)
Y = np.array([int(data[i][-1]) for i in range(1,len(data))]).reshape([len(data)-1])


X_train=X
Y_train=Y
#scaler = preprocessing.StandardScaler()
scaler = preprocessing.PowerTransformer()
scaler.fit(X_train)
#print(scaler.mean_)
X_train_trans = scaler.transform(X_train)
#X_test_trans = scaler.transform(X_test)

#pca = PCA()
pca = KernelPCA(kernel='rbf',n_components=20)

pca.fit(X_train_trans,Y_train)
pca_train = pca.transform(X_train_trans)
#pca_test = pca.transform(X_test_trans)
#print(pca_result)
#per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
#print(per_var)

'''
_, (train_ax, test_ax) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8, 4))
train_ax.scatter(pca_train[:, 0], pca_train[:, 1], c=Y_train)
#test_ax.scatter(pca_test[:, 0], pca_test[:, 1], c=Y_test)
plt.show()
'''

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(pca_train[:, 0], pca_train[:, 1], c=Y_train)
#ax=fig.add_subplot(1,2,2)
#ax.scatter(pca_test[:, 0], pca_test[:, 1], c=Y_test)
plt.savefig("pca_2d.png")

'''
fig=plt.figure()
ax=fig.add_subplot(1,1,1,projection="3d")
ax.scatter3D(pca_train[:, 0], pca_train[:, 1], pca_train[:, 2], c=Y_train)
#ax=fig.add_subplot(1,2,2,projection="3d")
#ax.scatter3D(pca_test[:, 0], pca_test[:, 1], pca_test[:, 2], c=Y_test)
plt.show()
'''
with open("pca_20.dat","w") as f:
    for i in range(len(X)):
        line=str(Y[i])
        for j in range(len(pca_train[i])):
            line+=" "+str(round(pca_train[i][j],5))
        line+="\n"
        f.write(line)
