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

# === 2. Spalten definieren ===
p_cols = ['p_human','p_landscape','p_building','p_edges','p_frequency','p_grayscale']
w_cols = ['w_human','w_landscape','w_building','w_edges','w_frequency','w_grayscale']

# === 3. Grundprüfung ===
print("=== Grundprüfung ===")
print("Label-Verteilung:", df['label'].value_counts(normalize=True).to_dict())
print("Prediction-Verteilung:", df['prediction'].value_counts(normalize=True).to_dict())
print("\nWertebereiche:")
for c in ['final_prob'] + p_cols + w_cols:
    print(f" {c:15s} -> {df[c].min():.3f} .. {df[c].max():.3f}")

# Interpretation
print("\n[Interpretation] Alle Werte sollten zwischen 0 und 1 liegen. "
      "Wenn Wahrscheinlichkeiten oder Gewichte außerhalb dieses Bereichs sind, "
      "liegt vermutlich ein Skalierungs- oder Normalisierungsfehler vor.\n")

# === 4. Gewichtssumme prüfen ===
df['w_sum'] = df[w_cols].sum(axis=1)
print("Gewichtssummen-Statistik:")
print(df['w_sum'].describe())

print("\n[Interpretation] Idealerweise sollte die Summe der Gewichte pro Zeile etwa 1 ergeben. "
      "Wenn sie stark schwankt oder deutlich von 1 abweicht, werden die Modelle unterschiedlich stark skaliert.\n")

plt.hist(df['w_sum'], bins=50)
plt.title("Summe der Gewichte pro Sample")
plt.xlabel("Summe der Gewichte")
plt.ylabel("Anzahl")
plt.show()

# === 5. Prüfen, ob final_prob korrekt berechnet wurde ===
weighted_sum = (df[p_cols].values * df[w_cols].values).sum(axis=1)
weight_sum = df[w_cols].sum(axis=1).replace(0, np.nan)
df['computed_final_prob'] = weighted_sum / weight_sum
df['final_diff'] = df['final_prob'] - df['computed_final_prob']

print("\nAbweichung zwischen final_prob und gewichteter Kombination:")
print(df['final_diff'].describe())

plt.hist(df['final_diff'].dropna(), bins=80)
plt.title("final_prob - computed_weighted_prob")
plt.xlabel("Differenz")
plt.ylabel("Anzahl")
plt.show()

print("\n[Interpretation] Wenn die Differenzen nahe 0 liegen, "
      "dann ist die Aggregation korrekt. Größere Abweichungen bedeuten, "
      "dass final_prob anders berechnet wird (z.B. Logits, Meta-Model, Rundungsfehler).\n")

# === 6. Modellqualität & Gewichtung ===
results = []
for p, w in zip(p_cols, w_cols):
    try:
        auc = roc_auc_score(df['label'], df[p])
    except:
        auc = np.nan
    acc = accuracy_score(df['label'], (df[p] >= 0.5).astype(int))
    mean_w = df[w].mean()
    results.append((p, auc, acc, mean_w))
res_df = pd.DataFrame(results, columns=['model','auc','acc','mean_weight'])
print("\nModellqualität & mittlere Gewichte:")
print(res_df.sort_values('auc', ascending=False))

print("\n[Interpretation] Modelle mit hoher AUC/Accuracy sollten tendenziell auch höhere Gewichte haben. "
      "Wenn das Gegenteil der Fall ist, liegt ein Fehler in der Gewichtungslogik vor.\n")

# === 7. Korrelation der Modellvorhersagen ===
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
mismatch = df[df['correct'] == 0]
mismatch['any_model_correct'] = ((mismatch[p_cols] >= 0.5).astype(int).eq(mismatch['label'], axis=0).sum(axis=1) > 0)
any_correct_ratio = mismatch['any_model_correct'].mean()
print(f"Fälle, bei denen Ensemble falsch war, aber mind. ein Modell richtig: {any_correct_ratio*100:.2f}%")

print("\n[Interpretation] Wenn dieser Anteil hoch ist, "
      "verliert die Gewichtung nützliche Signale. "
      "Ist er niedrig, liegen die Fehler eher an den Modellen selbst.\n")

# === 9. Kategorieanalyse (optional, falls img Struktur enthält) ===
def extract_category(path):
    parts = path.replace("\\", "/").split("/")
    # Nimm zweitletztes Element als Kategorie, z.B. .../landscape/img.jpg
    return parts[-2] if len(parts) >= 2 else "unknown"

df['category'] = df['img'].apply(extract_category)
cat_summary = df.groupby('category').apply(lambda g: pd.Series({
    'n': len(g),
    'acc': (g['prediction']==g['label']).mean(),
    'mean_final_prob': g['final_prob'].mean(),
    'mean_w_sum': g['w_sum'].mean()
}))
print("\nAccuracy & Gewichtssummen pro Kategorie:")
print(cat_summary)

print("\n[Interpretation] Große Unterschiede zwischen Kategorien können zeigen, "
      "dass die Gewichtsanpassung an Bildinhalt nicht gut funktioniert "
      "(z.B. zu niedriges w_human bei Human-Bildern).\n")

p_cols = ['p_human','p_landscape','p_building','p_edges','p_frequency','p_grayscale']
g1 = ['w_human','w_landscape','w_building']      # Inhalt
g2 = ['w_edges','w_frequency','w_grayscale']    # Qualität

# 1) Gruppen-Summe pro Sample
df['sum_g1'] = df[g1].sum(axis=1)
df['sum_g2'] = df[g2].sum(axis=1)
print("sum_g1 stats:", df['sum_g1'].describe())
print("sum_g2 stats:", df['sum_g2'].describe())

# 2) Korrelation Gruppen-Summe vs final_prob
print("corr sum_g1 vs final_prob:", df['sum_g1'].corr(df['final_prob']))
print("corr sum_g2 vs final_prob:", df['sum_g2'].corr(df['final_prob']))

# 3) Wie ändert sich AUC, wenn man nur Gruppe1 / nur Gruppe2 nutzt?
preds_g1 = (df[['p_human','p_landscape','p_building']].values * df[g1].values).sum(axis=1)
preds_g2 = (df[['p_edges','p_frequency','p_grayscale']].values * df[g2].values).sum(axis=1)
y = df['label'].values

print("AUC group1 (raw weighted):", roc_auc_score(y, preds_g1))
print("AUC group2 (raw weighted):", roc_auc_score(y, preds_g2))
print("AUC ensemble (final_prob):", roc_auc_score(y, df['final_prob']))


print("\n=== Diagnose abgeschlossen ===")
