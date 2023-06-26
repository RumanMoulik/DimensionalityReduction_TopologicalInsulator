import numpy as np
from sklearn.decomposition import KernelPCA, PCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from pickle import dump

with open("data.dat") as f:
    data = [line.strip().split() for line in f]
    
X=np.array([data[i][1:-1] for i in range(1,len(data))])
#print(np.shape(X))
samples, features = np.shape(X)
Y = np.array([int(data[i][-1]) for i in range(1,len(data))]).reshape([len(data)-1])

#Power Scaler Transformer
X_train=X
Y_train=Y
scaler = preprocessing.PowerTransformer()
scaler.fit(X_train)
X_train_trans = scaler.transform(X_train)

#Linear Kernel PCA
pca = KernelPCA(kernel='linear',n_components=20)
pca.fit(X_train_trans,Y_train)
pca_train = pca.transform(X_train_trans)
plt.figure()
plt.scatter(pca_train[:, 0], pca_train[:, 1], c=Y_train)
plt.savefig("pca/pca_linear.png")

#polynomial Kernel PCA
pca = KernelPCA(kernel='poly',n_components=20)
pca.fit(X_train_trans,Y_train)
pca_train = pca.transform(X_train_trans)
plt.figure()
plt.scatter(pca_train[:, 0], pca_train[:, 1], c=Y_train)
plt.savefig("pca/pca_polynomial.png")


#rbf Kernel PCA
pca = KernelPCA(kernel='rbf',n_components=20)
pca.fit(X_train_trans,Y_train)
pca_train = pca.transform(X_train_trans)
plt.figure()
plt.scatter(pca_train[:, 0], pca_train[:, 1], c=Y_train)
plt.savefig("pca/pca_rbf.png")

with open("pca_20.dat","w") as f:
    for i in range(len(X)):
        line=str(Y[i])
        for j in range(len(pca_train[i])):
            line+=" "+str(round(pca_train[i][j],5))
        line+="\n"
        f.write(line)

dump(scaler, open("trained_scaler.pkl",'wb'))
dump(pca, open("trained_pca.pkl",'wb'))
