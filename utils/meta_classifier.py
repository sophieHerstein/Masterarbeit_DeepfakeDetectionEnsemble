import csv
import os

from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, ParameterGrid, KFold, GridSearchCV
import pickle
import shutil


def test_for_best_classifier_train_data(test_size, all_table_keys):
    dataset = pd.read_csv("../logs/test/ensemble_1/ensemble_known_test_dir_details.csv")

    dataset = dataset.drop(['prediction', 'final_prob'], axis=1)
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

    lg_parameters = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2', None],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga'],
        'max_iter': [100, 200, 300],
        'class_weight': [None, 'balanced']
        }

    filtered_lg_parameters = create_lg_param_grid(lg_parameters)

    kfold = KFold(n_splits=5)
    grid_cv = GridSearchCV(estimator=LogisticRegression(), param_grid=filtered_lg_parameters, scoring='accuracy', cv=kfold)
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
        elif param_combination['penalty'] is None and param_combination['solver'] in ['newton-cg', 'lbfgs', 'saga']:
            filtered_param_grid.append(ensure_list(param_combination))
    return filtered_param_grid


def use_data_from_test_for_train_and_train_model():
    test_size = 0.7
    all_table_keys = False
    c = 0.01
    class_weight = None
    max_iter = 100
    penalty = None
    solver = 'lbfgs'

    train_data = pd.read_csv(f"train_{test_size}_{all_table_keys}.csv")
    test_data = pd.read_csv(f"test_{test_size}_{all_table_keys}.csv")

    train_data = train_data.drop(['img', 'w_human', 'w_landscape', 'w_building', 'w_edges', 'w_frequency', 'w_grayscale'], axis=1)
    test_data = test_data.drop(['img', 'w_human', 'w_landscape', 'w_building', 'w_edges', 'w_frequency', 'w_grayscale'], axis=1)

    X_train = train_data.drop(columns=['label'])
    y_train = train_data['label']
    X_test = test_data.drop(columns=['label'])
    y_test = test_data['label']

    lr_model = LogisticRegression(random_state=1, C=c, class_weight=class_weight, max_iter=max_iter, penalty=penalty, solver=solver)
    lr_model.fit(X_train, y_train)

    with open('../checkpoints/meta_classifier_for_ensemble.pkl', 'wb') as file:
        pickle.dump(lr_model, file)

    test_predictions = lr_model.predict(X_test)

    test_cm = confusion_matrix(y_test, test_predictions, labels=lr_model.classes_)

    print(test_cm)


def remove_train_images_from_test_for_ensemble_images():
    csv_path = "train_0.7_False.csv"  # deine CSV
    df = pd.read_csv(csv_path)

    source_root = "../data/test/known_test"
    target_root = "../data/test/meta_classifier_train_data"

    # CSV enthält z.B. eine Spalte "img" mit dem vollständigen Pfad
    for path in df["img"]:
        # Name der Datei extrahieren
        filename = os.path.basename(path)

        # Ordner (0_real oder 1_fake)
        label_dir = os.path.basename(os.path.dirname(path))

        # Quellpfad
        src = "../"+path

        # Zielpfad
        dst_dir = os.path.join(target_root, label_dir)
        os.makedirs(dst_dir, exist_ok=True)

        dst = os.path.join(dst_dir, filename)

        # Kopieren
        shutil.move(src, dst)

    print("Fertig! Bilder wurden kopiert.")


if __name__ == '__main__':
    # for el in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    #     test_for_best_classifier_train_data(el, False)
    #     test_for_best_classifier_train_data(el, True)
    # use_data_from_test_for_train_and_train_model()
    remove_train_images_from_test_for_ensemble_images()
