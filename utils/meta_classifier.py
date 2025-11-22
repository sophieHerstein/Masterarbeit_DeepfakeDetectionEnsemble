import csv
import os

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split, ParameterGrid, KFold, GridSearchCV
import pickle
import shutil


def test_for_best_classifier_train_data(all_table_keys, meta_values):
    train_df = pd.read_csv("train_meta.csv")

    filename = "meta_classifier_best_train_params.csv"
    if not os.path.exists(filename):
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "all_table_keys", "meta_values", "score", "params"])

    train_df = train_df.drop('img', axis=1)

    if not all_table_keys:
        train_df = train_df.drop(['w_human', 'w_landscape', 'w_building', 'w_edges', 'w_frequency', 'w_grayscale'], axis=1)

    if meta_values:
        prob_cols = ["p_human", "p_landscape", "p_building", "p_edges", "p_frequency", "p_grayscale"]

        train_df["conf_max"] = train_df[prob_cols].max(axis=1)
        train_df["conf_min"] = train_df[prob_cols].min(axis=1)
        train_df["conf_mean"] = train_df[prob_cols].mean(axis=1)
        train_df["conf_std"] = train_df[prob_cols].std(axis=1)

    X_train = train_df.drop(columns=['label'])
    y_train = train_df['label']

    lg_parameters = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2', None],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga'],
        'max_iter': [100, 200, 300],
        'class_weight': [None, 'balanced']
        }

    rfc_parameter = {
        'n_estimators': [50, 100, 200, 300],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 1, 5, 10, 'sqrt', 'log2'],
        'bootstrap': [True, False],
        'oob_score': [True, False]
    }

    gbc_parameters = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [1, 2, 3],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "subsample": [0.7, 0.9, 1.0]
    }

    filtered_lg_parameters = create_lg_param_grid(lg_parameters)
    filtered_rfc_parameters = create_rfc_param_grid(rfc_parameter)
    filtered_gbc_parameters = create_gbc_param_grid(gbc_parameters)

    model_list = [
        ('RFC', RandomForestClassifier(random_state=1), filtered_rfc_parameters),
        ('LR', LogisticRegression(random_state=1), filtered_lg_parameters),
        ('GBC', GradientBoostingClassifier(random_state=1), filtered_gbc_parameters),
    ]

    for name, model, parameters_for_testing in model_list:
        kfold = KFold(n_splits=5)
        grid_cv = GridSearchCV(estimator=model, param_grid=parameters_for_testing, scoring='accuracy', cv=kfold)
        result = grid_cv.fit(X_train, y_train)

        print("{}: Best {} using {}".format(name, result.best_score_, result.best_params_))

        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name, all_table_keys, meta_values, result.best_score_, result.best_params_])


def ensure_list(param_dict):
    for key in param_dict:
        if not isinstance(param_dict[key], list):
            param_dict[key] = [param_dict[key]]
    return param_dict

def create_lg_param_grid(params):
    param_grid = list(ParameterGrid(params))
    filtered_param_grid = []
    for param_combination in param_grid:
        if param_combination['penalty'] == 'l2' and param_combination['solver'] in ['newton-cg', 'lbfgs', 'liblinear', 'saga']:
            filtered_param_grid.append(ensure_list(param_combination))
        elif param_combination['penalty'] is None and param_combination['solver'] in ['newton-cg', 'lbfgs', 'saga']:
            filtered_param_grid.append(ensure_list(param_combination))
    return filtered_param_grid

def create_rfc_param_grid(params):
    param_grid = list(ParameterGrid(params))
    filtered_param_grid = []
    for param_combination in param_grid:
        if param_combination['oob_score'] == True and param_combination['bootstrap'] == False:
            continue
        filtered_param_grid.append(ensure_list(param_combination))
    return filtered_param_grid

def create_gbc_param_grid(params):
    param_grid = list(ParameterGrid(params))
    filtered = []
    for p in param_grid:
        # UNSINNIGE Kombinationen filtern:
        # subsample < 1.0 ist NUR sinnvoll bei n_estimators > 50
        if p["subsample"] < 1.0 and p["n_estimators"] < 100:
            continue
        filtered.append(ensure_list(p))
    return filtered

def use_data_from_test_for_train_and_train_model(all_table_keys, meta_values):
    filename = "meta_classifier_test_oder_so.csv"
    filename_unknown = "meta_classifier_unknown_test_oder_so.csv"
    if not os.path.exists(filename):
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "all_table_keys", "meta_values", "tn", "fp", "fn", "tp", "accuracy", "precision", "recall", "f1"])

    if not os.path.exists(filename_unknown):
        with open(filename_unknown, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "all_table_keys", "meta_values", "tn", "fp", "fn", "tp", "accuracy", "precision", "recall", "f1"])

    train_data = pd.read_csv(f"train_meta.csv")
    test_data = pd.read_csv(f"test_meta.csv")
    unknwon_test_data = pd.read_csv(f"test_meta_unknown.csv")

    train_data = train_data.drop(['img'], axis=1)
    test_data = test_data.drop(['img'], axis=1)
    unknwon_test_data = unknwon_test_data.drop(['img', 'prediction', 'final_prob'], axis=1)

    prob_cols = ["p_human", "p_landscape", "p_building", "p_edges", "p_frequency", "p_grayscale"]

    if meta_values:
        train_data["conf_max"] = train_data[prob_cols].max(axis=1)
        train_data["conf_min"] = train_data[prob_cols].min(axis=1)
        train_data["conf_mean"] = train_data[prob_cols].mean(axis=1)
        train_data["conf_std"] = train_data[prob_cols].std(axis=1)

        test_data["conf_max"] = test_data[prob_cols].max(axis=1)
        test_data["conf_min"] = test_data[prob_cols].min(axis=1)
        test_data["conf_mean"] = test_data[prob_cols].mean(axis=1)
        test_data["conf_std"] = test_data[prob_cols].std(axis=1)

        unknwon_test_data["conf_max"] = unknwon_test_data[prob_cols].max(axis=1)
        unknwon_test_data["conf_min"] = unknwon_test_data[prob_cols].min(axis=1)
        unknwon_test_data["conf_mean"] = unknwon_test_data[prob_cols].mean(axis=1)
        unknwon_test_data["conf_std"] = unknwon_test_data[prob_cols].std(axis=1)

    if not all_table_keys:
        train_data = train_data.drop(
            ['w_human', 'w_landscape', 'w_building', 'w_edges', 'w_frequency', 'w_grayscale'], axis=1)
        test_data = test_data.drop(
            ['w_human', 'w_landscape', 'w_building', 'w_edges', 'w_frequency', 'w_grayscale'], axis=1)
        unknwon_test_data = unknwon_test_data.drop(
                    ['w_human', 'w_landscape', 'w_building', 'w_edges', 'w_frequency', 'w_grayscale'], axis=1)


    X_train = train_data.drop(columns=['label'])
    y_train = train_data['label']
    X_test = test_data.drop(columns=['label'])
    y_test = test_data['label']
    X_unknown_test = unknwon_test_data.drop(columns=['label'])
    y_unknown_test = unknwon_test_data['label']

    if all_table_keys and meta_values:
        bootstrap = False
        criterion = 'entropy'
        max_depth = None
        max_features = 1
        min_samples_leaf = 2
        min_samples_split = 5
        n_estimators = 100
        oob_score = False

        rfc_model = RandomForestClassifier(random_state=1, bootstrap=bootstrap, criterion=criterion, max_depth=max_depth,
                                          max_features=max_features, min_samples_leaf=min_samples_leaf,
                                          min_samples_split=min_samples_split, n_estimators=n_estimators,
                                          oob_score=oob_score)
        rfc_model.fit(X_train, y_train)

        # with open('../checkpoints/meta_classifier_for_ensemble_with_weights.pkl', 'wb') as file:
        #     pickle.dump(lr_model, file)

        test_predictions = rfc_model.predict(X_test)
        tn, fp, fn, tp  = confusion_matrix(y_test, test_predictions, labels=rfc_model.classes_).ravel().tolist()

        report = classification_report(y_test, test_predictions, output_dict=True)

        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["RFC", all_table_keys, meta_values, tn, fp, fn, tp, report.get("accuracy"), report.get("1").get("precision"), report.get("1").get("recall"), report.get("1").get("f1-score")])


        unknown_test_predictions = rfc_model.predict(X_unknown_test)
        tn, fp, fn, tp = confusion_matrix(y_unknown_test, unknown_test_predictions, labels=rfc_model.classes_).ravel().tolist()

        report = classification_report(y_unknown_test, unknown_test_predictions, output_dict=True)

        with open(filename_unknown, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["RFC", all_table_keys, meta_values, tn, fp, fn, tp, report.get("accuracy"),
                             report.get("1").get("precision"), report.get("1").get("recall"),
                             report.get("1").get("f1-score")])
    elif all_table_keys:
        c = 0.01
        class_weight = None
        max_iter = 100
        penalty = None
        solver = "newton-cg"

        lr_model = LogisticRegression(random_state=1, C=c, class_weight=class_weight, max_iter=max_iter, penalty=penalty, solver=solver)
        lr_model.fit(X_train, y_train)

        # with open('../checkpoints/meta_classifier_for_ensemble_with_weights.pkl', 'wb') as file:
        #     pickle.dump(lr_model, file)

        test_predictions = lr_model.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, test_predictions, labels=lr_model.classes_).ravel().tolist()

        report = classification_report(y_test, test_predictions, output_dict=True)

        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["LR", all_table_keys, meta_values, tn, fp, fn, tp, report.get("accuracy"),
                             report.get("1").get("precision"), report.get("1").get("recall"),
                             report.get("1").get("f1-score")])

        unknown_test_predictions = lr_model.predict(X_unknown_test)
        tn, fp, fn, tp = confusion_matrix(y_unknown_test, unknown_test_predictions, labels=lr_model.classes_).ravel().tolist()

        report = classification_report(y_unknown_test, unknown_test_predictions, output_dict=True)

        with open(filename_unknown, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["LR", all_table_keys, meta_values, tn, fp, fn, tp, report.get("accuracy"),
                             report.get("1").get("precision"), report.get("1").get("recall"),
                             report.get("1").get("f1-score")])
    elif meta_values:
        bootstrap = True
        criterion = 'gini'
        max_depth = None
        max_features = 1
        min_samples_leaf = 2
        min_samples_split = 5
        n_estimators = 50
        oob_score = True

        rfc_model = RandomForestClassifier(random_state=1, bootstrap=bootstrap, criterion=criterion, max_depth=max_depth, max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, n_estimators=n_estimators, oob_score=oob_score)
        rfc_model.fit(X_train, y_train)

        # with open('../checkpoints/meta_classifier_for_ensemble_no_weights.pkl', 'wb') as file:
        #     pickle.dump(lr_model, file)

        test_predictions = rfc_model.predict(X_test)

        tn, fp, fn, tp = confusion_matrix(y_test, test_predictions, labels=rfc_model.classes_).ravel().tolist()

        report = classification_report(y_test, test_predictions, output_dict=True)

        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["RFC", all_table_keys, meta_values, tn, fp, fn, tp, report.get("accuracy"),
                             report.get("1").get("precision"), report.get("1").get("recall"),
                             report.get("1").get("f1-score")])

        unknwon_test_predictions = rfc_model.predict(X_unknown_test)
        tn, fp, fn, tp = confusion_matrix(y_unknown_test, unknwon_test_predictions, labels=rfc_model.classes_).ravel().tolist()

        report = classification_report(y_unknown_test, unknwon_test_predictions, output_dict=True)

        with open(filename_unknown, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["RFC", all_table_keys, meta_values, tn, fp, fn, tp, report.get("accuracy"),
                             report.get("1").get("precision"), report.get("1").get("recall"),
                             report.get("1").get("f1-score")])
    else:
        c = 0.01
        class_weight = None
        max_iter = 100
        penalty = None
        solver = "lbfgs"

        lr_model = LogisticRegression(random_state=1, C=c, class_weight=class_weight, max_iter=max_iter,
                                      penalty=penalty, solver=solver)
        lr_model.fit(X_train, y_train)

        # with open('../checkpoints/meta_classifier_for_ensemble_with_weights.pkl', 'wb') as file:
        #     pickle.dump(lr_model, file)

        test_predictions = lr_model.predict(X_test)

        tn, fp, fn, tp = confusion_matrix(y_test, test_predictions, labels=lr_model.classes_).ravel().tolist()

        report = classification_report(y_test, test_predictions, output_dict=True)

        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["LR", all_table_keys, meta_values, tn, fp, fn, tp, report.get("accuracy"),
                             report.get("1").get("precision"), report.get("1").get("recall"),
                             report.get("1").get("f1-score")])

        unknown_test_predictions = lr_model.predict(X_unknown_test)

        tn, fp, fn, tp = confusion_matrix(y_unknown_test, unknown_test_predictions, labels=lr_model.classes_).ravel().tolist()

        report = classification_report(y_unknown_test, unknown_test_predictions, output_dict=True)

        with open(filename_unknown, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["LR", all_table_keys, meta_values, tn, fp, fn, tp, report.get("accuracy"),
                             report.get("1").get("precision"), report.get("1").get("recall"),
                             report.get("1").get("f1-score")])


def remove_train_images_from_test_for_ensemble_images():
    csv_path = "train_meta.csv"  # deine CSV
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

def test_meta_classifier():
    lr_model = pickle.load(open('../checkpoints/meta_classifier_for_ensemble_with_weights.pkl', 'rb'))
    test_data = pd.read_csv("test_meta.csv")
    test_data = test_data.drop(['img', 'label'], axis=1)
    predictions = lr_model.predict_proba(test_data)
    print(predictions[0][1])

if __name__ == '__main__':
    # test_for_best_classifier_train_data(False, True)
    # test_for_best_classifier_train_data(True, True)
    # test_for_best_classifier_train_data(False, False)
    # test_for_best_classifier_train_data(True, False)
    use_data_from_test_for_train_and_train_model(True, True)
    use_data_from_test_for_train_and_train_model(False, True)
    use_data_from_test_for_train_and_train_model(True, False)
    use_data_from_test_for_train_and_train_model(False, False)
    # remove_train_images_from_test_for_ensemble_images()
    # test_meta_classifier()
