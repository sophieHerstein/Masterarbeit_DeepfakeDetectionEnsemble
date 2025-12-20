import csv
import os

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split, ParameterGrid, KFold, GridSearchCV
import pickle
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent  # Pfad zu diesem Skript
PROJECT_DIR = BASE_DIR.parent

def test_for_best_classifier_train_data(all_table_keys, meta_values, meta_type):
    train_df = pd.read_csv(f"../../logs/meta_classifier/train/train_meta_{meta_type}.csv")

    filename = f"../../logs/meta_classifier/meta_classifier_best_train_params_{meta_type}.csv"
    if not os.path.exists(filename):
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "all_table_keys", "meta_values", "score", "params"])

    if meta_type == "base":
        train_df = train_df.drop(['img', "img_norm", "img_id"], axis=1)
    else:
        train_df = train_df.drop(['img', 'final_prob', 'prediction'], axis=1)

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
        'C': [0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['newton-cg', 'lbfgs'],
        'max_iter': [100, 200, 300],
        'class_weight': [None, 'balanced']
        }

    rfc_parameter = {
        'n_estimators': [100, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True],
        'oob_score': [True]
    }

    gbc_parameters = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3],
        "subsample": [0.7, 1.0]
    }

    filtered_lg_parameters = create_lg_param_grid(lg_parameters)
    filtered_rfc_parameters = create_rfc_param_grid(rfc_parameter)
    filtered_gbc_parameters = create_gbc_param_grid(gbc_parameters)

    model_list = [
        ('RFC', RandomForestClassifier(random_state=1, n_jobs=1), filtered_rfc_parameters),
        ('LR', LogisticRegression(random_state=1, n_jobs=1), filtered_lg_parameters),
        ('GBC', GradientBoostingClassifier(random_state=1), filtered_gbc_parameters),
    ]

    for name, model, parameters_for_testing in model_list:
        kfold = KFold(n_splits=5)
        grid_cv = GridSearchCV(estimator=model, param_grid=parameters_for_testing, scoring='accuracy', cv=kfold, n_jobs=-1, verbose=2, error_score='raise')
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

def create_file(filename):
    if not os.path.exists(filename):
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "all_table_keys", "meta_values", "tn", "fp", "fn", "tp", "accuracy", "precision", "recall", "f1"])


def test_model(model_name, model, X_test, y_test, filename, all_table_keys, meta_values):
    test_predictions = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, test_predictions, labels=model.classes_).ravel().tolist()

    report = classification_report(y_test, test_predictions, output_dict=True)

    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([model_name, all_table_keys, meta_values, tn, fp, fn, tp, report.get("accuracy"),
                         report.get("1").get("precision"), report.get("1").get("recall"),
                         report.get("1").get("f1-score")])


def use_data_from_test_for_train_and_train_model(all_table_keys, meta_values, meta_type):
    # filename = "../../logs/meta_classifier/test_results/meta_classifier_test.csv"
    # filename_unknown = "../../logs/meta_classifier/test_results/meta_classifier_unknown_test.csv"
    # filename_jpeg = "../../logs/meta_classifier/test_results/meta_classifier_jpeg_test.csv"
    # filename_noisy = "../../logs/meta_classifier/test_results/meta_classifier_noisy_test.csv"
    # filename_scaled = "../../logs/meta_classifier/test_results/meta_classifier_scaled_test.csv"
    #
    # create_file(filename)
    # create_file(filename_unknown)
    # create_file(filename_jpeg)
    # create_file(filename_noisy)
    # create_file(filename_scaled)

    train_data = pd.read_csv(f"../../logs/meta_classifier/train/train_meta_{meta_type}.csv")
    # test_data = pd.read_csv(f"../../logs/meta_classifier/test_data/base/test_meta.csv")
    # unknwon_test_data = pd.read_csv(f"../test_meta_unknown.csv")
    # jpeg_test_data = pd.read_csv(f"../test_meta_jpeg.csv")
    # noisy_test_data = pd.read_csv(f"../test_meta_noisy.csv")
    # scaled_test_data = pd.read_csv(f"../test_meta_scaled.csv")

    if meta_type == "base":
        train_data = train_data.drop(['img', 'img_norm', 'img_id'], axis=1)
    else:
        train_data = train_data.drop(['img', 'final_prob', 'prediction'], axis=1)
    # noisy_test_data = noisy_test_data.drop(['img', 'img_norm', 'img_id'], axis=1)
    # jpeg_test_data = jpeg_test_data.drop(['img', 'img_norm', 'img_id'], axis=1)
    # scaled_test_data = scaled_test_data.drop(['img', 'img_norm', 'img_id'], axis=1)
    # test_data = test_data.drop(['img'], axis=1)
    # unknwon_test_data = unknwon_test_data.drop(['img', 'prediction', 'final_prob'], axis=1)

    prob_cols = ["p_human", "p_landscape", "p_building", "p_edges", "p_frequency", "p_grayscale"]

    if meta_values:
        train_data["conf_max"] = train_data[prob_cols].max(axis=1)
        train_data["conf_min"] = train_data[prob_cols].min(axis=1)
        train_data["conf_mean"] = train_data[prob_cols].mean(axis=1)
        train_data["conf_std"] = train_data[prob_cols].std(axis=1)

        # test_data["conf_max"] = test_data[prob_cols].max(axis=1)
        # test_data["conf_min"] = test_data[prob_cols].min(axis=1)
        # test_data["conf_mean"] = test_data[prob_cols].mean(axis=1)
        # test_data["conf_std"] = test_data[prob_cols].std(axis=1)
        #
        # unknwon_test_data["conf_max"] = unknwon_test_data[prob_cols].max(axis=1)
        # unknwon_test_data["conf_min"] = unknwon_test_data[prob_cols].min(axis=1)
        # unknwon_test_data["conf_mean"] = unknwon_test_data[prob_cols].mean(axis=1)
        # unknwon_test_data["conf_std"] = unknwon_test_data[prob_cols].std(axis=1)
        #
        # noisy_test_data["conf_max"] = noisy_test_data[prob_cols].max(axis=1)
        # noisy_test_data["conf_min"] = noisy_test_data[prob_cols].min(axis=1)
        # noisy_test_data["conf_mean"] = noisy_test_data[prob_cols].mean(axis=1)
        # noisy_test_data["conf_std"] = noisy_test_data[prob_cols].std(axis=1)
        #
        # jpeg_test_data["conf_max"] = jpeg_test_data[prob_cols].max(axis=1)
        # jpeg_test_data["conf_min"] = jpeg_test_data[prob_cols].min(axis=1)
        # jpeg_test_data["conf_mean"] = jpeg_test_data[prob_cols].mean(axis=1)
        # jpeg_test_data["conf_std"] = jpeg_test_data[prob_cols].std(axis=1)
        #
        # scaled_test_data["conf_max"] = scaled_test_data[prob_cols].max(axis=1)
        # scaled_test_data["conf_min"] = scaled_test_data[prob_cols].min(axis=1)
        # scaled_test_data["conf_mean"] = scaled_test_data[prob_cols].mean(axis=1)
        # scaled_test_data["conf_std"] = scaled_test_data[prob_cols].std(axis=1)

    if not all_table_keys:
        train_data = train_data.drop(
            ['w_human', 'w_landscape', 'w_building', 'w_edges', 'w_frequency', 'w_grayscale'], axis=1)
        # test_data = test_data.drop(
        #     ['w_human', 'w_landscape', 'w_building', 'w_edges', 'w_frequency', 'w_grayscale'], axis=1)
        # unknwon_test_data = unknwon_test_data.drop(
        #             ['w_human', 'w_landscape', 'w_building', 'w_edges', 'w_frequency', 'w_grayscale'], axis=1)
        # noisy_test_data = noisy_test_data.drop(
        #             ['w_human', 'w_landscape', 'w_building', 'w_edges', 'w_frequency', 'w_grayscale'], axis=1)
        # jpeg_test_data = jpeg_test_data.drop(
        #             ['w_human', 'w_landscape', 'w_building', 'w_edges', 'w_frequency', 'w_grayscale'], axis=1)
        # scaled_test_data = scaled_test_data.drop(
        #             ['w_human', 'w_landscape', 'w_building', 'w_edges', 'w_frequency', 'w_grayscale'], axis=1)


    X_train = train_data.drop(columns=['label'])
    y_train = train_data['label']
    # X_test = test_data.drop(columns=['label'])
    # y_test = test_data['label']
    # X_unknown_test = unknwon_test_data.drop(columns=['label'])
    # y_unknown_test = unknwon_test_data['label']
    # X_noisy_test = noisy_test_data.drop(columns=['label'])
    # y_noisy_test = noisy_test_data['label']
    # X_scaled_test = scaled_test_data.drop(columns=['label'])
    # y_scaled_test = scaled_test_data['label']
    # X_jpeg_test = jpeg_test_data.drop(columns=['label'])
    # y_jpeg_test = jpeg_test_data['label']

    if all_table_keys and meta_values:
        learning_rate = 0.1
        max_depth = 3
        n_estimators = 200
        subsample = 0.7

        gbc_model = GradientBoostingClassifier(random_state=1, learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators, subsample=subsample)
        gbc_model.fit(X_train, y_train)

        with open(f'../../checkpoints/meta_classifier_for_ensemble_{meta_type}_with_weights.pkl', 'wb') as file:
            pickle.dump(gbc_model, file)

        # test_model("GBC", gbc_model, X_test, y_test, filename, all_table_keys, meta_values)
        # test_model("GBC", gbc_model, X_unknown_test, y_unknown_test, filename_unknown, all_table_keys, meta_values)
        # test_model("GBC", gbc_model, X_scaled_test, y_scaled_test, filename_scaled, all_table_keys, meta_values)
        # test_model("GBC", gbc_model, X_jpeg_test, y_jpeg_test, filename_jpeg, all_table_keys, meta_values)
        # test_model("GBC", gbc_model, X_noisy_test, y_noisy_test, filename_noisy, all_table_keys, meta_values)
    elif all_table_keys:
        bootstrap = True
        max_depth = None
        max_features = 'sqrt'
        min_samples_leaf = 1
        min_samples_split = 5
        n_estimators = 100
        oob_score = True

        rfc_model = RandomForestClassifier(random_state=1, bootstrap=bootstrap, n_estimators=n_estimators, max_features=max_features, max_depth=max_depth, min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split, oob_score=oob_score)
        rfc_model.fit(X_train, y_train)

        with open(f'../../checkpoints/meta_classifier_for_ensemble_{meta_type}_with_weights.pkl', 'wb') as file:
            pickle.dump(rfc_model, file)

        # test_model("RFC", rfc_model, X_test, y_test, filename, all_table_keys, meta_values)
        # test_model("RFC", rfc_model, X_unknown_test, y_unknown_test, filename_unknown, all_table_keys, meta_values)
        # test_model("RFC", rfc_model, X_scaled_test, y_scaled_test, filename_scaled, all_table_keys, meta_values)
        # test_model("RFC", rfc_model, X_jpeg_test, y_jpeg_test, filename_jpeg, all_table_keys, meta_values)
        # test_model("RFC", rfc_model, X_noisy_test, y_noisy_test, filename_noisy, all_table_keys, meta_values)
    elif meta_values:
        bootstrap = True
        max_depth = 10
        max_features = 'sqrt'
        min_samples_leaf = 1
        oob_score = True
        min_samples_split = 2
        n_estimators = 100

        rfc_model = RandomForestClassifier(random_state=1, bootstrap=bootstrap, max_depth=max_depth, max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, n_estimators=n_estimators, oob_score=oob_score)
        rfc_model.fit(X_train, y_train)

        with open(f'../../checkpoints/meta_classifier_for_ensemble_{meta_type}_no_weights.pkl', 'wb') as file:
            pickle.dump(rfc_model, file)

        # test_model("RFC", rfc_model, X_test, y_test, filename, all_table_keys, meta_values)
        # test_model("RFC", rfc_model, X_unknown_test, y_unknown_test, filename_unknown, all_table_keys, meta_values)
        # test_model("RFC", rfc_model, X_scaled_test, y_scaled_test, filename_scaled, all_table_keys, meta_values)
        # test_model("RFC", rfc_model, X_jpeg_test, y_jpeg_test, filename_jpeg, all_table_keys, meta_values)
        # test_model("RFC", rfc_model, X_noisy_test, y_noisy_test, filename_noisy, all_table_keys, meta_values)
    else:
        bootstrap = True
        max_features = 'sqrt'
        min_samples_leaf = 2
        oob_score = True
        n_estimators = 100

        if meta_type == "not_specialized":
            max_depth = 20
            min_samples_split = 2

        else:
            max_depth = 10
            min_samples_split = 5

        rfc_model = RandomForestClassifier(random_state=1, bootstrap=bootstrap, max_depth=max_depth, max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, n_estimators=n_estimators, oob_score=oob_score)
        rfc_model.fit(X_train, y_train)

        with open(f'../../checkpoints/meta_classifier_for_ensemble_{meta_type}_no_weights.pkl', 'wb') as file:
            pickle.dump(rfc_model, file)

        # test_model("RFC", rfc_model, X_test, y_test, filename, all_table_keys, meta_values)
        # test_model("RFC", rfc_model, X_unknown_test, y_unknown_test, filename_unknown, all_table_keys, meta_values)
        # test_model("RFC", rfc_model, X_scaled_test, y_scaled_test, filename_scaled, all_table_keys, meta_values)
        # test_model("RFC", rfc_model, X_jpeg_test, y_jpeg_test, filename_jpeg, all_table_keys, meta_values)
        # test_model("RFC", rfc_model, X_noisy_test, y_noisy_test, filename_noisy, all_table_keys, meta_values)


def remove_train_images_from_test_for_ensemble_images():
    csv_path = "logs/meta_classifier/train/train_meta_base.csv"
    df = pd.read_csv(csv_path)

    target_root = "../data/test/meta_classifier_train_data"

    for path in df["img"]:
        filename = os.path.basename(path)

        label_dir = os.path.basename(os.path.dirname(path))

        src = "../"+path

        dst_dir = os.path.join(target_root, label_dir)
        os.makedirs(dst_dir, exist_ok=True)

        dst = os.path.join(dst_dir, filename)

        shutil.move(src, dst)

    print("Fertig! Bilder wurden kopiert.")


def normalize(path: str) -> str:
    """Normalize path separators + lowercase."""
    return path.replace("\\", "/").lower()


VARIANTS = {"jpeg", "noisy", "scaled"}

def extract_base_id(path: str) -> str:
    path = normalize(path)
    filename = Path(path).stem.lower()  # architecture_1750_noisy
    parts = filename.split("_")

    # Letzter Teil ist nur Varianten-Suffix?
    if parts[-1] in VARIANTS:
        return "_".join(parts[:-1])

    # Sonst NICHT abschneiden!
    return filename



def get_relative_subpath(path: str) -> Path:
    """
    Gibt den Pfad relativ zu data/test/... zurück.
    Normalisiert Backslashes und Case.
    """
    p = Path(normalize(path))
    parts = p.parts
    idx = parts.index("test") + 1  # alles nach 'test/'
    return Path(*parts[idx:])


# ---------------------------------------
# Hauptlogik
# ---------------------------------------

def get_train_images_for_robustheit():

    base = pd.read_csv("logs/meta_classifier/train/train_meta_base.csv")
    base["img_norm"] = base["img"].apply(normalize)
    base["img_id"] = base["img_norm"].apply(extract_base_id)

    variants = ["jpeg", "noisy", "scaled"]

    appended_rows = [base]  # Start mit Base

    variant_dfs = {}  # später fürs Kopieren gebraucht

    for var in variants:
        df = pd.read_csv(f"../logs/test/ensemble/weighted_ensemble_known_test_{var}_dir_details.csv")

        # Entferne unnötige Spalten
        df = df.drop(["prediction", "final_prob"], axis=1)

        df["img_norm"] = df["img"].apply(normalize)
        df["img_id"] = df["img_norm"].apply(extract_base_id)

        variant_dfs[var] = df  # speichern

        # Match + Missing
        matched = df[df["img_id"].isin(base["img_id"])]
        missing = df[~df["img_id"].isin(base["img_id"])]

        # Missing speichern
        missing.to_csv(f"test_meta_{var}.csv", index=False)

        appended_rows.append(matched)

        print(f"[{var}] matched: {len(matched)}, missing: {len(missing)}")

    # ---------------------------------------
    # Neue Base schreiben
    # ---------------------------------------
    new_base = pd.concat(appended_rows, ignore_index=True)
    new_base.to_csv("train_meta_merged.csv", index=False)

    print("\nNeue Base-Gesamtgröße:", len(new_base))


    # ---------------------------------------
    # Bilder kopieren
    # ---------------------------------------
    TARGET_ROOT = Path("../data/test/meta_classifier_train_data")

    for var in variants:
        df = variant_dfs[var]

        # Jetzt MATCH gegen die neue Base (wichtiger Fix!)
        matched = df[df["img_id"].isin(new_base["img_id"])]

        for img_path in matched["img"]:
            src = PROJECT_DIR / normalize(img_path)

            # Sicherstellen, dass die Datei existiert
            if not src.exists():
                print(f"⚠ WARNUNG: Datei existiert nicht: {src}")
                continue

            relative = get_relative_subpath(normalize(img_path))   # bleibt Variante erhalten!
            dest = TARGET_ROOT / relative

            os.makedirs(dest.parent, exist_ok=True)

            shutil.move(src, dest)

        print(f"[{var}] kopiert: {len(matched)} Bilder")

    print("\nFERTIG ✓")


def test_meta_classifier():
    lr_model = pickle.load(open('../../checkpoints/meta_classifier_for_ensemble_base_with_weights.pkl', 'rb'))
    test_data = pd.read_csv("logs/meta_classifier/test_data/base/test_meta.csv")
    test_data = test_data.drop(['img', 'label'], axis=1)
    predictions = lr_model.predict_proba(test_data)
    print(predictions[0][1])

if __name__ == '__main__':
    # get_train_images_for_robustheit()
    # for t in ["base", "diverse", "not_specialized"]:
    #     test_for_best_classifier_train_data(False, True, t)
    #     test_for_best_classifier_train_data(False, False, t)
    #     if t == "not_specialized":
    #         continue
    #     test_for_best_classifier_train_data(True, True, t)
    #     test_for_best_classifier_train_data(True, False, t)
    # use_data_from_test_for_train_and_train_model(True, True, "base")
    # use_data_from_test_for_train_and_train_model(False, True, "base")
    # use_data_from_test_for_train_and_train_model(True, False, "diverse")
    # use_data_from_test_for_train_and_train_model(False, False, "diverse")
    use_data_from_test_for_train_and_train_model(False, False, "not_specialized")
    # remove_train_images_from_test_for_ensemble_images()
    # test_meta_classifier()
