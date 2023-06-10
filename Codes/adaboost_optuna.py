import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_validate
import optuna
from optuna.samplers import TPESampler

with open("pca_20.dat") as f:
    data = [line.strip().split() for line in f]
    
dim = len(data[0])-1

X=np.array([data[i][1:] for i in range(len(data))])
Y = np.array([int(data[i][0]) for i in range(len(data))]).reshape([len(data)])

def objective(trial):
    trees = trial.suggest_int("n_estimators",50,2000)
    lr = trial.suggest_float("learning_rate",0.001,0.3, log=True)
    clf = AdaBoostClassifier(n_estimators=trees, learning_rate=lr)
    cvf = cross_validate(clf,X,Y,cv=5)
    return max(cvf['test_score'])
    
sampler = TPESampler(seed=1)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=100)

print("N_trials=",len(study.trials))
print("Best TrialValue:",study.best_trial.value)
print("Params:")
for key,value in study.best_trial.params.items():
    print(" {}: {}".format(key,value))
