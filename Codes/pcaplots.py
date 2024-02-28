import numpy as np
from sklearn.decomposition import KernelPCA, PCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from pickle import dump

file_names = ["grp_62-site_12","grp_139-site_5","grp_189-site_9","grp_221-site_2","grp_221-site_5","grp_225-site_2"]

for i in file_names:
    with open("..//"+i+"//data.dat") as f:
        data = [line.strip().split() for line in f]
        
    X=np.array([data[i][1:-1] for i in range(1,len(data))])
    #print(np.shape(X))
    samples, features = np.shape(X)
    Y = np.array([int(data[i][-1]) for i in range(1,len(data))]).reshape([len(data)-1])
    fig = plt.figure(figsize=(6.5,1.7))

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
    ax = fig.add_subplot(1,3,1,projection='3d')
    ax.scatter(pca_train[:, 0], pca_train[:, 1], pca_train[:, 2], s=3, c=Y_train)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    #polynomial Kernel PCA
    pca = KernelPCA(kernel='poly',n_components=20)
    pca.fit(X_train_trans,Y_train)
    pca_train = pca.transform(X_train_trans)
    ax = fig.add_subplot(1,3,2,projection='3d')
    ax.scatter(pca_train[:, 0], pca_train[:, 1], pca_train[:, 2], s=3, c=Y_train)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


    #rbf Kernel PCA
    pca = KernelPCA(kernel='rbf',n_components=20)
    pca.fit(X_train_trans,Y_train)
    pca_train = pca.transform(X_train_trans)
    ax = fig.add_subplot(1,3,3,projection='3d')
    ax.scatter(pca_train[:, 0], pca_train[:, 1], pca_train[:, 2], s=3, c=Y_train)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.show()
    
    fig.tight_layout()
    plt.show()
    fig.savefig("pca_"+i+".pdf", format='pdf')