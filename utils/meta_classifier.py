import csv
import os

from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split, ParameterGrid, KFold, GridSearchCV

dataset = pd.read_csv("../logs/test/ensemble_1/ensemble_known_test_dir_details.csv")

dataset = dataset.drop(['prediction', 'final_prob'], axis=1)

def test_for_best_classifier_train_data(test_size, all_table_keys):
    filename = "meta_classifier_best_train_params.csv"
    if not os.path.exists(filename):
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["test_size", "all_table_keys", "score", "params"])

    train_df, test_df = train_test_split(
        dataset,
        test_size=test_size,
        random_state=42,
        stratify=dataset["label"]
    )

    train_df.to_csv(f"train_{test_size}_{all_table_keys}.csv", index=False)
    test_df.to_csv(f"test_{test_size}_{all_table_keys}.csv", index=False)

    train_df = train_df.drop('img', axis=1)
    test_df = test_df.drop('img', axis=1)

    if not all_table_keys:
        train_df = train_df.drop(['w_human', 'w_landscape', 'w_building', 'w_edges', 'w_frequency', 'w_grayscale'], axis=1)
        test_df = test_df.drop(['w_human', 'w_landscape', 'w_building', 'w_edges', 'w_frequency', 'w_grayscale'], axis=1)

    X_train = train_df.drop(columns=['label'])
    y_train = train_df['label']
    X_test = test_df.drop(columns=['label'])
    y_test = test_df['label']

    lg_parameters = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2', None],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga'],
        'max_iter': [100, 200, 300],
        'class_weight': [None, 'balanced']
        }

    filtered_lg_parameters = create_lg_param_grid(lg_parameters)

    kfold = KFold(n_splits=5)
    grid_cv = GridSearchCV(estimator=LogisticRegression, param_grid=filtered_lg_parameters, scoring='accuracy', cv=kfold)
    result = grid_cv.fit(X_train, y_train)

    print("Best {} using {}".format(result.best_score_, result.best_params_))

    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([test_size, all_table_keys, result.best_score_, result.best_params_])


def ensure_list(param_dict):
    for key in param_dict:
        if not isinstance(param_dict[key], list):
            param_dict[key] = [param_dict[key]]
    return param_dict

# Filterfunktion für Logistic Regression
def create_lg_param_grid(params):
    param_grid = list(ParameterGrid(params))
    filtered_param_grid = []
    for param_combination in param_grid:
        if param_combination['penalty'] == 'l2' and param_combination['solver'] in ['newton-cg', 'lbfgs', 'liblinear', 'saga']:
            filtered_param_grid.append(ensure_list(param_combination))
        elif param_combination['penalty'] == 'none' and param_combination['solver'] in ['newton-cg', 'lbfgs', 'saga']:
            filtered_param_grid.append(ensure_list(param_combination))
    return filtered_param_grid


if __name__ == '__main__':
    for el in [0.2, 0.3, 0.4, 0.5]:
        test_for_best_classifier_train_data(el, False)
        test_for_best_classifier_train_data(el, True)
