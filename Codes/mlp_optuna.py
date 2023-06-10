import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
import optuna
from optuna.samplers import TPESampler

with open("pca_20.dat") as f:
    data = [line.strip().split() for line in f]
    
dim = len(data[0])-1

X=np.array([data[i][1:] for i in range(len(data))],dtype=float)
Y = np.array([int(data[i][0]) for i in range(len(data))],dtype=int).reshape([len(data)])

def objective(trial):
    iters = trial.suggest_int("max_iterations",1000,4000)
    lr = trial.suggest_float("learning_rate_init",0.0001,0.01, log=True)
    alpha = trial.suggest_float("alpha",0.00001,0.001, log=True)
    hl_size = trial.suggest_int("hidden layer size",50,500, log=True)
    hl_depth = trial.suggest_int("hidden layer depth",1,10, log=True)
    hl = np.full(hl_depth,hl_size)
    clf = MLPClassifier(hidden_layer_sizes=hl,max_iter=iters, learning_rate_init=lr, alpha=alpha)
    cvf = cross_validate(clf,X,Y,cv=5)
    return max(cvf['test_score'])
    
sampler = TPESampler(seed=1)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=10000)

print("N_trials=",len(study.trials))
print("Best TrialValue:",study.best_trial.value)
print("Params:")
for key,value in study.best_trial.params.items():
    print(" {}: {}".format(key,value))
