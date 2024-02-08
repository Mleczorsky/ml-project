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
            max_depth = trial.suggest_int('max_depth', 2, 30)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 40)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
            max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 3, 30)
            classifier = DecisionTreeClassifier(max_depth=max_depth,
                                                min_samples_split=min_samples_split,
                                                min_samples_leaf=min_samples_leaf,
                                                max_leaf_nodes=max_leaf_nodes)
        elif model_name == 'random_forest':
            n_estimators = trial.suggest_int('n_estimators', 100, 1000)
            max_depth = trial.suggest_int('max_depth', 2, 70)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 40)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
            max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 3, 70)
            classifier = RandomForestClassifier(n_estimators=n_estimators,
                                                max_depth=max_depth,
                                                min_samples_split=min_samples_split,
                                                min_samples_leaf=min_samples_leaf,
                                                max_leaf_nodes=max_leaf_nodes,
                                                n_jobs=-1)
        elif model_name == 'svm':
            C = trial.suggest_float('C', 0.05, 100)
            kernel = trial.suggest_categorical('kernel', ['poly'])
            gamma = trial.suggest_float('gamma', 1e-5, 1.0)
            degree = trial.suggest_int('degree', 2, 9)
            classifier = SVC(C=C, kernel=kernel, gamma=gamma, degree=degree)

        elif model_name == 'xgboost':
            max_depth = trial.suggest_int('max_depth', 3, 12)
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2)
            subsample = trial.suggest_float('subsample', 0.5, 1.0)
            reg_alpha = trial.suggest_float('reg_alpha', 0, 5)
            reg_lambda = trial.suggest_float('reg_lambda', 0, 5)
            classifier = XGBClassifier(max_depth=max_depth,
                                       n_estimators=n_estimators,
                                       learning_rate=learning_rate,
                                       subsample=subsample,
                                       reg_alpha=reg_alpha,
                                       reg_lambda=reg_lambda)

        score = cross_val_score(classifier, X, y, n_jobs=-1, cv=10)
        return score.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)
    return study.best_trial
