"""diverse Plots für Auswertung erzeugen"""
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.config import TRAININGS_VARIANTEN, TEST_VARIANTEN, ALL_MODELS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_test_results(model):
    path = f"logs/test/{model}_metrics.csv"
    if os.path.exists(os.path.join(PROJECT_ROOT, path)):
        return pd.read_csv(os.path.join(PROJECT_ROOT, path))
    return None


def get_train_plots():
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "plots", "train_comparison")
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs", "train")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for variante in TRAININGS_VARIANTEN:
        log_files = glob(os.path.join(LOG_DIR, variante, "*.csv"))

        all_dfs = []
        for file in log_files:
            if not file.endswith("search.csv") and not file.endswith("results.csv"):
                try:
                    df = pd.read_csv(file)
                    df["Epoche"] = df["Epoche"].astype(int)
                    df["Modell"] = get_model_name(os.path.basename(file).replace(".csv", ""))
                    df["Variante"] = os.path.basename(os.path.dirname(file))
                    all_dfs.append(df)
                except Exception as e:
                    print(f"❌ Fehler beim Laden von {file}: {e}")

        if not all_dfs:
            print("⚠️ Keine Trainingslogs gefunden.")
            exit()

        df_all = pd.concat(all_dfs, ignore_index=True)

        metric = "Val-Acc"
        varianten = df_all["Variante"].unique()

        for variante in varianten:
            df_subset = df_all[df_all["Variante"] == variante]

            plt.figure(figsize=(8, 5))

            sns.lineplot(
                data=df_subset,
                x="Epoche",
                y=metric,
                hue="Modell",
                marker="o",
                linewidth=2
            )

            for modell in df_subset["Modell"].unique():
                df_mod = df_subset[df_subset["Modell"] == modell]

                max_idx = df_mod[metric].idxmax()
                x_max = df_mod.loc[max_idx, "Epoche"]
                y_max = df_mod.loc[max_idx, metric]

                plt.scatter(
                    x_max, y_max,
                    s=75,
                    zorder=5
                )
                ax = plt.gca()

                colors = {line.get_label(): line.get_color() for line in ax.lines}
                line_color = colors[modell]

                plt.plot([x_max, x_max], [df_subset[metric].min(), y_max], linestyle="--", alpha=0.4, color=line_color)
                plt.plot([df_subset["Epoche"].min(), x_max], [y_max, y_max], linestyle="--", alpha=0.4,
                         color=line_color)

            plt.title(f"Validation Accuracy über Epochen – {get_category(variante)}")
            max_epoch = df_subset["Epoche"].max()
            plt.xticks(range(0, max_epoch + 1, 2))
            plt.xlabel("Epochen")
            plt.ylabel("Validation Accuracy")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()

            filename = f"{variante}-{metric.lower().replace('-', '_')}.svg"
            plt.savefig(os.path.join(OUTPUT_DIR, filename))
            plt.close()

        print(f"✅ Plots gespeichert in {OUTPUT_DIR}")


def get_confusion_matrices():
    for model in ALL_MODELS:
        for testvariante in TEST_VARIANTEN:
            OUTPUT_DIR = os.path.join(PROJECT_ROOT, "plots", "confusion_matrices")

            filename = f"{model}_{testvariante}_confusion_matrix.svg"
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            test_df = load_test_results(model)
            df = test_df.loc[test_df['TestVariante'] == testvariante]

            cm = np.array([[df["TN"].iloc[0], df["FN"].iloc[0]],
                           [df["FP"].iloc[0], df["TP"].iloc[0]]])

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"], ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"Confusion Matrix ({get_model_name(model)})")
            plt.savefig(os.path.join(OUTPUT_DIR, filename))
            plt.close()

            print(f"✅ Plots gespeichert in {OUTPUT_DIR}")


def get_test_plots():
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "plots", "test_comparison")
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs", "test")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for variante in TEST_VARIANTEN:

        data = []
        for model in ALL_MODELS:
            file = os.path.join(LOG_DIR, f"{model}_metrics.csv")
            df = pd.read_csv(file)
            df = df.loc[df['TestVariante'] == variante]
            if df.empty:
                continue

            data.append({
                "Model": model,
                "Accuracy": df["Accuracy"].iloc[0],
                "Precision": df["Precision"].iloc[0],
                "Recall": df["Recall"].iloc[0],
                "F1-Score": df["F1-Score"].iloc[0],
                "ROC-AUC": df["ROC-AUC"].iloc[0],
                "TP": df["TN"].iloc[0]
            })

        if variante in ["known_test_insertion", "unknown_test_insertion"]:
            data = sorted(data, key=lambda x: x["TP"], reverse=True)
        else:
            data = sorted(data, key=lambda x: x["Accuracy"], reverse=True)

        models = [get_model_name(d["Model"]) for d in data]
        if variante in ["known_test_insertion", "unknown_test_insertion"]:
            tp = [d["TP"] for d in data]
        else:
            accuracies = [d["Accuracy"] for d in data]
            precisions = [d["Precision"] for d in data]
            recalls = [d["Recall"] for d in data]
            f1_scores = [d["F1-Score"] for d in data]
            roc_aucs = [d["ROC-AUC"] for d in data]

        if variante in ["known_test_insertion", "unknown_test_insertion"]:
            x = np.arange(len(models))

            fig, ax = plt.subplots(figsize=(10, 6))
            width = 0.5
            ax.bar(x, tp, width, label="True Positives")

            ax.set_title(f"Modellvergleich ({get_variante(variante)})")
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=90, )
            ax.yaxis.set_major_locator(plt.MultipleLocator(20))
            ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.set_axisbelow(True)
            plt.tight_layout()

            output_path = os.path.join(OUTPUT_DIR, f"{variante}_comparison.svg")
            plt.savefig(output_path, dpi=300)
            plt.show()
            print(f"✅ Vergleichsplot gespeichert unter: {output_path}")
        else:
            x = np.arange(len(models))
            width = 0.15

            fig, ax = plt.subplots(figsize=(10, 6))

            ax.bar(x - 1.5 * width, accuracies, width, label="Accuracy")
            ax.bar(x - 0.5 * width, precisions, width, label="Precision")
            ax.bar(x + 0.5 * width, recalls, width, label="Recall")
            ax.bar(x + 1.5 * width, f1_scores, width, label="F1-Score")
            ax.bar(x + 2.5 * width, roc_aucs, width, label="ROC-AUC")

            ax.set_title(f"Modellvergleich ({get_variante(variante)})")
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=90)
            ax.set_ylim(0, 1.0)
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
            ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.set_axisbelow(True)
            ax.legend(ncol=2, fontsize=10, loc="lower right")
            plt.tight_layout()

            output_path = os.path.join(OUTPUT_DIR, f"{variante}_comparison.svg")
            plt.savefig(output_path, dpi=300)
            plt.show()
            print(f"✅ Vergleichsplot gespeichert unter: {output_path}")


def get_model_name(model):
    if model == "xception71":
        return "Xception"
    if model == "mobilenetv2_100":
        return "MobileNetV2"
    if model == "tf_efficientnet_b3":
        return "EfficientNet"
    if model == "densenet121":
        return "DenseNet"
    if model == "resnet50d":
        return "ResNet"
    if model == "convnext_small":
        return "ConvNext"
    if model == "weighted_ensemble":
        return "gewichtetes Ensemble"
    if model == "unweighted_ensemble":
        return "ungewichtetes Ensemble"
    if model == "weighted_meta_classifier_ensemble":
        return "gewichtetes Ensemble \nmit Meta Classifier"
    if model == "unweighted_meta_classifier_ensemble":
        return "ungewichtetes Ensemble \nmit Meta Classifier"
    if model == "weighted_ensemble_diverse":
        return "gewichtetes Ensemble \nmit diversen Detektoren"
    if model == "unweighted_ensemble_diverse":
        return "ungewichtetes Ensemble \nmit diversen Detektoren"
    if model == "weighted_meta_classifier_ensemble_diverse":
        return "gewichtetes Ensemble \nmit diversen Detektoren \nund Meta Classifier"
    if model == "unweighted_meta_classifier_ensemble_diverse":
        return "ungewichtetes Ensemble \nmit diversen Detektoren \nund Meta Classifier"
    if model == "not_specialized_ensemble":
        return "nicht-spezialisiertes Ensemble"
    if model == "not_specialized_meta_classifier_ensemble":
        return "nicht-spezialisiertes Ensemble \nmit Meta Classifier"
    return "NOT FOUND"


def get_category(category):
    if category == "human":
        return "Mensch"
    elif category == "building":
        return "Gebäude"
    elif category == "landscape":
        return "Landschaft"
    elif category == "frequencies":
        return "Frequenz"
    elif category == "edges":
        return "Kanten"
    elif category == "grayscaling":
        return "Graustufen"
    elif category == "single_models":
        return "Einzelmodelle"
    return "UNKNOWN"


def get_variante(variante):
    if variante == "known_test_dir":
        return "In-Dataset-Tests"
    elif variante == "unknown_test_dir":
        return "Cross-Dataset-Tests"
    elif variante == "known_test_jpeg_dir":
        return "In-Dataset-Tests mit komprimierten Bildern"
    elif variante == "unknown_test_jpeg_dir":
        return "Cross-Dataset-Tests mit komprimierten Bildern"
    elif variante == "known_test_noisy_dir":
        return "In-Dataset-Tests mit verrauschten Bildern"
    elif variante == "unknown_test_noisy_dir":
        return "Cross-Dataset-Tests mit verrauschten Bildern"
    elif variante == "known_test_scaled_dir":
        return "In-Dataset-Tests mit skalierten Bildern"
    elif variante == "unknown_test_scaled_dir":
        return "Cross-Dataset-Tests mit skalierten Bildern"
    elif variante == "known_test_occlusion":
        return "In-Dataset-Tests mit Okklusion"
    elif variante == "unknown_test_occlusion":
        return "Cross-Dataset-Tests mit Okklusion"
    elif variante == "known_test_insertion":
        return "In-Dataset-Tests mit Fake Insertion"
    elif variante == "unknown_test_insertion":
        return "Cross-Dataset-Tests mit Fake Insertion"
    elif variante == "known_test_format_png":
        return "In-Dataset-Tests mit Bildern im png-Format"
    elif variante == "unknown_test_format_png":
        return "Cross-Dataset-Tests mit Bildern im png-Format"
    elif variante == "known_test_format_webp":
        return "In-Dataset-Tests mit Bildern im webp-Format"
    elif variante == "unknown_test_format_webp":
        return "Cross-Dataset-Tests mit Bildern im webp-Format"
    elif variante == "known_test_histogram":
        return "In-Dataset-Tests mit Histogrammänderungen"
    elif variante == "unknown_test_histogram":
        return "Cross-Dataset-Tests mit Histogrammänderungen"
    return "UNKNOWN"


def load_acc(model, variante):
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "plots", "poster")
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs", "test")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file = os.path.join(LOG_DIR, f"{model}_metrics.csv")
    if not os.path.exists(file):
        return None
    df = pd.read_csv(file)
    df = df.loc[df["TestVariante"] == variante]
    if df.empty:
        return None
    return df["Accuracy"].iloc[0]


def get_plot_for_poster():
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "plots", "poster")
    data = []

    VAR_NORMAL_KNOWN = "known_test_dir"
    VAR_NORMAL_UNKNOWN = "unknown_test_dir"
    VAR_JPEG_KNOWN = "known_test_jpeg_dir"
    VAR_JPEG_UNKNOWN = "unknown_test_jpeg_dir"
    VAR_SCALED_KNOWN = "known_test_scaled_dir"
    VAR_SCALED_UNKNOWN = "unknown_test_scaled_dir"
    VAR_NOISY_KNOWN = "known_test_noisy_dir"
    VAR_NOISY_UNKNOWN = "unknown_test_noisy_dir"

    for m in ALL_MODELS:
        nk = load_acc(m, VAR_NORMAL_KNOWN)
        nu = load_acc(m, VAR_NORMAL_UNKNOWN)
        jk = load_acc(m, VAR_JPEG_KNOWN)
        ju = load_acc(m, VAR_JPEG_UNKNOWN)
        sk = load_acc(m, VAR_SCALED_KNOWN)
        su = load_acc(m, VAR_SCALED_UNKNOWN)
        nok = load_acc(m, VAR_NOISY_KNOWN)
        nou = load_acc(m, VAR_NOISY_UNKNOWN)

        if None in [nk, nu, jk, ju, sk, su, nok, nou]:
            continue

        data.append({
            "model": m,
            "normal_known": nk,
            "normal_unknown": nu,
            "jpeg_known": jk,
            "jpeg_unknown": ju,
            "scaled_known": sk,
            "scaled_unknown": su,
            "noisy_known": nok,
            "noisy_unknown": nou
        })

    data = sorted(data, key=lambda x: x["normal_known"], reverse=True)

    models = [get_model_name(d["model"]) for d in data]
    x = np.arange(len(models))
    width = 0.15

    known_color = "#FF4A70"
    unknown_color = "#E40139"
    known_jpeg_color = "#B6002E"
    unknown_jpeg_color = "#FF7A95"
    known_scaled_color = "#8C0024"
    unknown_scaled_color = "#FF9DB2"
    known_noisy_color = "#600019"
    unknown_noisy_color = "#FFC4D0"

    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)

    offset_normal = -1.5 * width
    offset_jpeg = -0.5 * width
    offset_scaled = 0.5 * width
    offset_noisy = 1.5 * width

    ax.bar(
        x + offset_normal,
        [d["normal_known"] for d in data],
        width,
        label="Known (normal)",
        color=known_color
    )
    ax.bar(
        x + offset_normal,
        [d["normal_unknown"] for d in data],
        width,
        label="Unknown (normal)",
        color=unknown_color
    )

    ax.bar(
        x + offset_jpeg,
        [d["jpeg_known"] for d in data],
        width,
        label="Known (jpeg)",
        color=known_jpeg_color
    )
    ax.bar(
        x + offset_jpeg,
        [d["jpeg_unknown"] for d in data],
        width,
        label="Unknown (jpeg)",
        color=unknown_jpeg_color
    )

    ax.bar(
        x + offset_scaled,
        [d["scaled_known"] for d in data],
        width,
        label="Known (scaled)",
        color=known_scaled_color
    )
    ax.bar(
        x + offset_scaled,
        [d["scaled_unknown"] for d in data],
        width,
        label="Unknown (scaled)",
        color=unknown_scaled_color
    )

    ax.bar(
        x + offset_noisy,
        [d["noisy_known"] for d in data],
        width,
        label="Known (noisy)",
        color=known_noisy_color
    )
    ax.bar(
        x + offset_noisy,
        [d["noisy_unknown"] for d in data],
        width,
        label="Unknown (noisy)",
        color=unknown_noisy_color
    )

    ax.set_title("Accuracy – Known vs. Unknown (Normal & verändert)", fontsize=16)
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=90)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    ax.legend(ncol=2, fontsize=10, loc="lower right")
    plt.tight_layout()

    output = os.path.join(OUTPUT_DIR, "overlay_known_unknown.svg")
    plt.savefig(output, dpi=300)
    plt.show()

    print(f"Plot gespeichert unter: {output}")


def get_robustness_plot():
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "plots", "robustness_comparison")
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs", "test")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

    for variante in TEST_VARIANTEN:
        data = []

        for model in ALL_MODELS:
            file = os.path.join(LOG_DIR, f"{model}_metrics.csv")
            df = pd.read_csv(file)

            if "unknown" in variante and variante != "unknown_test_dir":
                base_name = "unknown_test_dir"
            else:
                base_name = "known_test_dir"

            base_row = df.loc[df["TestVariante"] == base_name]
            var_row = df.loc[df["TestVariante"] == variante]

            if base_row.empty or var_row.empty:
                continue

            base_row = base_row.iloc[0]
            var_row = var_row.iloc[0]

            deltas = {m: var_row[m] - base_row[m] for m in metrics}

            entry = {
                "Model": model,
                "Δ_Accuracy": deltas["Accuracy"],
                "Δ_Precision": deltas["Precision"],
                "Δ_Recall": deltas["Recall"],
                "Δ_F1": deltas["F1-Score"],
                "Δ_ROC": deltas["ROC-AUC"],
                "Accuracy": var_row["Accuracy"]
            }

            data.append(entry)

        if not data:
            continue

        data = sorted(data, key=lambda x: x["Accuracy"], reverse=True)

        models = [get_model_name(d["Model"]) for d in data]
        d_acc = [d["Δ_Accuracy"] for d in data]
        d_pre = [d["Δ_Precision"] for d in data]
        d_rec = [d["Δ_Recall"] for d in data]
        d_f1 = [d["Δ_F1"] for d in data]
        d_roc = [d["Δ_ROC"] for d in data]

        x = np.arange(len(models))
        width = 0.15

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - 1.5 * width, d_acc, width, label="Δ Accuracy")
        ax.bar(x - 0.5 * width, d_pre, width, label="Δ Precision")
        ax.bar(x + 0.5 * width, d_rec, width, label="Δ Recall")
        ax.bar(x + 1.5 * width, d_f1, width, label="Δ F1-Score")
        ax.bar(x + 2.5 * width, d_roc, width, label="Δ ROC-AUC")

        ax.axhline(0, color='black', linestyle='--', linewidth=1)

        ax.set_title(f"Robustheitsvergleich ({get_variante(variante)})")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=90)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)
        ax.legend(ncol=2, fontsize=10)

        plt.tight_layout()

        output_path = os.path.join(OUTPUT_DIR, f"{variante}_robustness.svg")
        plt.savefig(output_path, dpi=300)
        plt.show()

        print(f"✅ Robustness Comparison Plot gespeichert unter: {output_path}")


def get_zusammenfassungs_plot():
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "plots", "test_comparison")
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs", "test")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = []

    for model in ALL_MODELS:
        sums = {
            "Accuracy": 0.0,
            "Precision": 0.0,
            "Recall": 0.0,
            "F1-Score": 0.0,
            "ROC-AUC": 0.0,
        }
        count = 0

        file = os.path.join(LOG_DIR, f"{model}_metrics.csv")
        df = pd.read_csv(file)

        for variante in TEST_VARIANTEN:
            if variante in ["known_test_insertion", "unknown_test_insertion"]:
                continue

            row = df.loc[df["TestVariante"] == variante]
            if row.empty:
                continue

            for metric in sums:
                sums[metric] += row[metric].iloc[0]

            count += 1

        if count == 0:
            continue

        data.append({
            "Model": model,
            **{metric: value / count for metric, value in sums.items()}
        })

    data = sorted(data, key=lambda x: x["Accuracy"], reverse=True)

    models = [get_model_name(d["Model"]) for d in data]

    accuracies = [d["Accuracy"] for d in data]
    precisions = [d["Precision"] for d in data]
    recalls = [d["Recall"] for d in data]
    f1_scores = [d["F1-Score"] for d in data]
    roc_aucs = [d["ROC-AUC"] for d in data]

    x = np.arange(len(models))
    width = 0.15

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - 1.5 * width, accuracies, width, label="Accuracy")
    ax.bar(x - 0.5 * width, precisions, width, label="Precision")
    ax.bar(x + 0.5 * width, recalls, width, label="Recall")
    ax.bar(x + 1.5 * width, f1_scores, width, label="F1-Score")
    ax.bar(x + 2.5 * width, roc_aucs, width, label="ROC-AUC")

    ax.set_title(f"Modellvergleich (Zusammenfassung)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=90)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, fontsize=10, loc="lower right")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, f"zusammenfassung.svg")
    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f"✅ Vergleichsplot gespeichert unter: {output_path}")


if __name__ == "__main__":
    # get_train_plots()
    # get_confusion_matrices()
    get_test_plots()
    # get_plot_for_poster()
    get_robustness_plot()
    get_zusammenfassungs_plot()
