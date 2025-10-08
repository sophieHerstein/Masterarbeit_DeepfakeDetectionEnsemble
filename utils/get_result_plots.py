import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from utils.config import TRAININGS_VARIANTEN, MODELS, TEST_VARIANTEN

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
        # === Alle Logdateien laden ===
        log_files = glob(os.path.join(LOG_DIR, variante, "*.csv"))

        all_dfs = []
        for file in log_files:
            if not file.endswith("search.csv") and not file.endswith("results.csv"):
                try:
                    df = pd.read_csv(file)
                    df["Epoche"] = df["Epoche"].astype(int)
                    df["Modell"] = os.path.basename(file).replace(".csv", "")
                    df["Variante"] = os.path.basename(os.path.dirname(file))
                    all_dfs.append(df)
                except Exception as e:
                    print(f"❌ Fehler beim Laden von {file}: {e}")

        if not all_dfs:
            print("⚠️ Keine Trainingslogs gefunden.")
            exit()

        # === Alle Daten zusammenfassen ===
        df_all = pd.concat(all_dfs, ignore_index=True)

        # === Plotten ===
        metrics = ["Loss", "Train-Acc", "Val-Acc"]
        varianten = df_all["Variante"].unique()

        for metric in metrics:
            for variante in varianten:
                df_subset = df_all[df_all["Variante"] == variante]
                plt.figure(figsize=(8, 5))
                sns.lineplot(data=df_subset, x="Epoche", y=metric, hue="Modell", marker="o")
                plt.title(f"{metric} über Epochen – {variante}")
                max_epoch = df_subset["Epoche"].max()
                plt.xticks(range(0, max_epoch + 1, 2))
                plt.xlabel("Epoche")
                plt.ylabel(metric)
                plt.grid(True, linestyle="--", alpha=0.6)
                plt.tight_layout()
                filename = f"{variante}-{metric.lower().replace('-', '_')}.png"
                plt.savefig(os.path.join(OUTPUT_DIR, filename))
                plt.close()

        print(f"✅ Plots gespeichert in {OUTPUT_DIR}")

def get_confusion_matrices():
    for model in [*MODELS, "ensemble", "unweighted_ensemble"]:
        for testvariante in TEST_VARIANTEN:
            OUTPUT_DIR = os.path.join(PROJECT_ROOT, "plots", "confusion_matrices")

            filename = f"{model}_{testvariante}_confusion_matrix.png"
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
            ax.set_title("Confusion Matrix")
            plt.savefig(os.path.join(OUTPUT_DIR, filename))
            plt.close()

            print(f"✅ Plots gespeichert in {OUTPUT_DIR}")


def get_test_plots():
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "plots", "test_comparison")
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs", "test")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for variante in TEST_VARIANTEN:
        # === Daten sammeln ===
        models = []
        accuracies, precisions, recalls, f1_scores, roc_aucs = [], [], [], [], []

        for model in [*MODELS, "ensemble", "unweighted_ensemble"]:
            file = os.path.join(LOG_DIR, f"{model}_metrics.csv")
            df = pd.read_csv(file)
            df = df.loc[df['TestVariante'] == variante]
            if df.empty:
                continue
            models.append(model)
            accuracies.append(df["Accuracy"].iloc[0])
            precisions.append(df["Precision"].iloc[0])
            recalls.append(df["Recall"].iloc[0])
            f1_scores.append(df["F1-Score"].iloc[0])
            roc_aucs.append(df["ROC-AUC"].iloc[0])

        # === Plot vorbereiten ===
        x = np.arange(len(models))
        width = 0.2

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.bar(x - 1.5 * width, accuracies, width, label="Accuracy")
        ax.bar(x - 0.5 * width, precisions, width, label="Precision")
        ax.bar(x + 0.5 * width, recalls, width, label="Recall")
        ax.bar(x + 1.5 * width, f1_scores, width, label="F1-Score")

        # === Achsen und Beschriftung ===
        ax.set_title(f"Modellvergleich ({variante})")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        plt.tight_layout()

        # === Speichern und Anzeigen ===
        output_path = os.path.join(OUTPUT_DIR, f"{variante}_comparison.png")
        plt.savefig(output_path)
        plt.show()
        print(f"✅ Vergleichsplot gespeichert unter: {output_path}")


if __name__ == "__main__":
    get_train_plots()
    get_confusion_matrices()
    get_test_plots()