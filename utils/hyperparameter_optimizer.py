import optuna
from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC


def optimize_hyperparameters(model_name : str, X, y, n_trials=100):
    """
    Optimize hyperparameters by maximizing mean accuracy measured by cross-validation

    `model_name` should be one of: `'knn'`, `'decision_tree'`, `'random_forest'`, `'svm'`, `'xgboost'`
    """

    def objective(trial):
        '''returns mean model accuracy measured by cross-validation'''

        if model_name == 'knn':
            n_neighbors = trial.suggest_int('n_neighbors', 3, 49)
            classifier = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        elif model_name == 'decision_tree':
            ...
        elif model_name == 'random_forest':
            ...
        elif model_name == 'svm':
            ...
        elif model_name == 'xgboost':
            ...

        score = cross_val_score(classifier, X, y, n_jobs=-1, cv=10)
        return score.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)
    return study.best_trial
