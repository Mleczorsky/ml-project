import pickle

def unpickle( name ):
    with open(f'/pickled_models/{ name }.pkl', 'rb') as file:
        return pickle.load(file)

decision_tree = unpickle('best_decision_tree_model_trained_on_augmented_data')
knn = unpickle('best_knn_model_trained_on_augmented_data')
random_forest = unpickle('best_random_forest_model_trained_on_augmented_data')
svm = unpickle('best_svm_model_trained_on_augmented_data')
xgboost = unpickle('best_xgboost_model_trained_on_augmented_data')

models = {
    "decision_tree": decision_tree,
    "knn": knn,
    "random_forest": random_forest,
    "svm": svm,
    "xgboost": xgboost
}