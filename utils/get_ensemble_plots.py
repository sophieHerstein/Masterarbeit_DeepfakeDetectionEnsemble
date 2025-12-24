import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from utils.config import TRAININGS_VARIANTEN, TEST_VARIANTEN, ALL_MODELS
from utils.get_result_plots import get_model_name

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "plots", "ensemble_results")
p_cols = ['p_human', 'p_landscape', 'p_building', 'p_edges', 'p_frequency', 'p_grayscale']
w_cols = ['w_human', 'w_landscape', 'w_building', 'w_edges', 'w_frequency', 'w_grayscale']

def load_ensemble_results(ensemble_type, test_type):
    path = f"logs/test/ensemble/extra_column/{ensemble_type}_{test_type}_details.csv"
    if os.path.exists(os.path.join(PROJECT_ROOT, path)):
        return pd.read_csv(os.path.join(PROJECT_ROOT, path))
    return None


def get_plots_for_klassentrennung(ensembles, test_dir):
    for ensemble in ensembles:
        ensemble_name = get_model_name(ensemble)
        df = load_ensemble_results(ensemble, test_dir)
        if df is None:
            print(f"❌ Plot nicht möglich für {ensemble}")
        real = df[df["label"] == 0]["final_prob"]
        fake = df[df["label"] == 1]["final_prob"]

        plt.figure()
        plt.hist(real, bins=30, alpha=0.6, label="Real")
        plt.hist(fake, bins=30, alpha=0.6, label="Fake")
        plt.axvline(0.5, linestyle="--")
        plt.xlabel("final_prob")
        plt.ylabel("Anzahl")
        plt.title(f"{ensemble_name} – Trennung der Klassen")
        plt.legend()
        output_path = os.path.join(OUTPUT_DIR, f"{ensemble}_{test_dir}_klassentrennung.svg")
        plt.savefig(output_path)
        plt.show()
        print(f"✅ Plot Trennung der Klassen gespeichert unter: {output_path}")


def get_plots_for_native_model_errors(test_dir):
    CATEGORY_TO_MODEL = {
        "human": "p_human",
        "building": "p_building",
        "landscape": "p_landscape",
    }

    df = load_ensemble_results("unweighted_ensemble", test_dir)
    if df is None:
        print(f"❌ Plot nicht möglich für unweighted_ensemble")

    fp_counts = []
    fn_counts = []

    for cat, p_col in CATEGORY_TO_MODEL.items():
        sub = df[df["category"] == cat]
        if sub.empty:
            fp_counts.append(0)
            fn_counts.append(0)
            continue

        pred = (sub[p_col].astype(float) >= 0.5).astype(int)
        label = sub["label"].astype(int)

        fp = ((pred == 1) & (label == 0)).sum()
        fn = ((pred == 0) & (label == 1)).sum()

        fp_counts.append(int(fp))
        fn_counts.append(int(fn))

    x = np.arange(len(CATEGORY_TO_MODEL))
    width = 0.6

    plt.figure()
    plt.bar(x, fp_counts, width, label="False Positives")
    plt.bar(x, fn_counts, width, bottom=fp_counts, label="False Negatives")

    plt.xticks(x, ["Mensch", "Gebäude", "Landschaft"])
    plt.ylabel("Anzahl Fehlklassifikationen")
    plt.title("FP/FN der Kategorie-Classifier auf ihrer Kategorie")
    plt.legend()

    output_path = os.path.join(
        OUTPUT_DIR,
            f"{test_dir}_kategorie_native_fp_fn_stacked.svg"
        )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

    print(f"✅ Plot FP/FN gestapelt gespeichert unter: {output_path}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # get_plots_for_klassentrennung(["weighted_ensemble", "unweighted_ensemble"], "known_test_dir")
    # get_plots_for_klassentrennung(["weighted_ensemble", "unweighted_ensemble", "weighted_meta_classifier_ensemble", "unweighted_meta_classifier_ensemble"], "unknown_test_jpeg_dir")
    # get_plots_for_klassentrennung(["weighted_ensemble", "unweighted_ensemble", "weighted_meta_classifier_ensemble", "unweighted_meta_classifier_ensemble"], "unknown_test_noisy_dir")
    get_plots_for_native_model_errors("unknown_test_noisy_dir")
    get_plots_for_native_model_errors("unknown_test_jpeg_dir")
