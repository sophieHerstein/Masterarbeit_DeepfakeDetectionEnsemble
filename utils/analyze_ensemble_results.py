import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, accuracy_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(PROJECT_ROOT, "logs/test/ensemble/ensemble_unknown_test_dir_details.csv")

df = pd.read_csv(path)

print("\n=== Datensatz geladen ===")
print(f"Samples: {len(df)} | Spalten: {len(df.columns)}\n")

p_cols = ['p_human','p_landscape','p_building','p_edges','p_frequency','p_grayscale']
w_cols = ['w_human','w_landscape','w_building','w_edges','w_frequency','w_grayscale']

print("=== Grundprüfung ===")
print("Label-Verteilung:", df['label'].value_counts(normalize=True).to_dict())
print("Prediction-Verteilung:", df['prediction'].value_counts(normalize=True).to_dict())
print("\nWertebereiche:")
for c in ['final_prob'] + p_cols + w_cols:
    print(f" {c:15s} -> {df[c].min():.3f} .. {df[c].max():.3f}")


plt.figure(figsize=(8,6))
corr = df[p_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=0, vmax=1)
plt.title("Korrelation der Modellwahrscheinlichkeiten")
plt.show()

print("[Interpretation] Sehr hohe Korrelationen (>0.9) bedeuten, dass die Modelle sehr ähnlich entscheiden "
      "und das Ensemble kaum Diversität bringt. Niedrige Werte zeigen komplementäre Modelle, "
      "was grundsätzlich gut für Ensembles ist.\n")

# === 8. Fehleranalyse ===
df['correct'] = (df['prediction'] == df['label']).astype(int)
acc_ensemble = df['correct'].mean()
print(f"Gesamt-Accuracy Ensemble: {acc_ensemble:.3f}")

# Fälle, bei denen Ensemble falsch, aber mind. ein Modell richtig
mismatch = df[df['correct'] == 0].copy()
mismatch['any_model_correct'] = ((mismatch[p_cols] >= 0.5).astype(int).eq(mismatch['label'], axis=0).sum(axis=1) > 0)
any_correct_ratio = mismatch['any_model_correct'].mean()
print(f"Fälle, bei denen Ensemble falsch war, aber mind. ein Modell richtig: {any_correct_ratio*100:.2f}%")

print("\n[Interpretation] Wenn dieser Anteil hoch ist, "
      "verliert die Gewichtung nützliche Signale. "
      "Ist er niedrig, liegen die Fehler eher an den Modellen selbst.\n")

print("\n=== Diagnose abgeschlossen ===")
