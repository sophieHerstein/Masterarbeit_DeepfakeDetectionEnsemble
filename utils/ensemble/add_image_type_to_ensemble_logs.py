"""nachträgliche Ergänzung der Bildkategorien in Ensemble Logs"""
import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = f"../logs/test/ensemble"


def classify_image(title: str):
    title = title.lower()

    if "landscape" in title:
        return "landscape"
    elif "building" in title:
        return "building"
    elif "human" in title:
        return "human"
    elif "lhq" in title:
        return "landscape"
    elif "architecture" in title:
        return "building"
    elif "faceforensics" in title:
        return "human"
    elif "celeba" in title:
        return "human"
    elif "ffhq" in title:
        return "human"
    elif "imagenet" in title:
        return "building"

    else:
        return "UNKNOWN"


def add_column(df: pd.DataFrame):
    df["category"] = df["img"].apply(classify_image)
    return df


def add_column_for_all_csvs():
    folder = Path(os.path.join(PROJECT_ROOT, path))
    for csv_path in folder.glob("*.csv"):
        print(csv_path)
        df = pd.read_csv(csv_path)
        df_out = add_column(df)
        print(df_out)
        out_path = os.path.join(PROJECT_ROOT, path, "extra_column", csv_path.name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df_out.to_csv(out_path, index=False)


if __name__ == '__main__':
    add_column_for_all_csvs()
