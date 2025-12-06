import csv
import os
import pandas as pd

from utils.config import TRAININGS_VARIANTEN, MODELS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs", "train")

for train_variante in TRAININGS_VARIANTEN:
    train_variante_dir = os.path.join(LOG_DIR, train_variante)

    for model in MODELS:
        if os.path.exists(os.path.join(train_variante_dir, f"{model}.csv")):
            df = pd.read_csv(os.path.join(train_variante_dir, f"{model}.csv"))
        else:
            continue

        best_val_acc = df.loc[df["Val-Acc"].idxmax()]

        out_file = os.path.join(train_variante_dir, "final_train_results.csv")
        log_exists = os.path.isfile(out_file)
        with open(out_file, "a", newline="") as logfile:
            writer = csv.writer(logfile)
            if not log_exists:
                writer.writerow(["Modell", "Train-Acc", "Val-Acc", "Loss", "Eoche"])
            writer.writerow([
                model,
                best_val_acc["Train-Acc"],
                best_val_acc["Val-Acc"],
                best_val_acc["Loss"],
                best_val_acc["Epoche"],
            ])
