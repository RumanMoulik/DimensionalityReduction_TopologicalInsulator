import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import KernelPCA, PCA

mlp = pickle.load(open("../trained_mlp.pkl","rb"))
pca = pickle.load(open("../trained_pca.pkl","rb"))
scaler = pickle.load(open("../trained_scaler.pkl","rb"))

with open("test.dat") as f:
    test = [line.strip().split() for line in f]
    
X = np.array([test[i][1:-2] for i in range(1,len(test))], dtype=float)
#print(np.shape(X))
samples, features = np.shape(X)
topo = np.array([int(test[i][-2]) for i in range(1,len(test))]).reshape([len(test)-1])
mp_ids = np.array([(test[i][-1]) for i in range(1,len(test))]).reshape([len(test)-1])
formulae = np.array([(test[i][0]) for i in range(1,len(test))]).reshape([len(test)-1])

X_trans = scaler.transform(X)
X_pca = pca.transform(X_trans)
X_pca_slice = X_pca[:,:4]
Y_mlp = mlp.predict(X_pca_slice)

f = open("validation.dat","w")
width = 12
print("formula".ljust(width)+"mp_id".ljust(width)+"data".ljust(width)+"prediction".ljust(width))

for i in range(len(X)):
    if (X[i][3]+X[i][4]+X[i][5]) %2 == 0:
        print(formulae[i].ljust(width) + mp_ids[i].ljust(width) + str(topo[i]).ljust(width) + str(Y_mlp[i]).ljust(width) + "1")
    else:
        print(formulae[i].ljust(width) + mp_ids[i].ljust(width) + str(topo[i]).ljust(width) + str(Y_mlp[i]).ljust(width) + "0")
    
    if topo[i] == -1 and Y_mlp[i] == 1 and (X[i][3]+X[i][4]+X[i][5]) %2 == 0:
        f.write(mp_ids[i]+"\n")

f.close()
