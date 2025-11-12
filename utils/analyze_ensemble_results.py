import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(PROJECT_ROOT, "logs/test/ensemble/unweighted_ensemble_unknown_test_dir_details.csv")

df = pd.read_csv(path)

print("\n=== Datensatz geladen ===")

# === 2. Spalten definieren ===
p_cols = ['p_human','p_landscape','p_building','p_edges','p_frequency','p_grayscale']
w_cols = ['w_human','w_landscape','w_building','w_edges','w_frequency','w_grayscale']

not_as_deepfake_identified_images = df[df['prediction']==0]
fn = not_as_deepfake_identified_images[not_as_deepfake_identified_images['label']==1].copy()
fn['num_of_correct_models'] =(fn[p_cols] >= 0.5).sum(axis=1)

p_human_right_for_deepfake = (fn['p_human'] >= 0.5).sum()
p_landscape_right_for_deepfake = (fn['p_landscape'] >= 0.5).sum()
p_building_right_for_deepfake = (fn['p_building'] >= 0.5).sum()
p_edges_right_for_deepfake = (fn['p_edges'] >= 0.5).sum()
p_frequency_right_for_deepfake = (fn['p_frequency'] >= 0.5).sum()
p_grayscale_right_for_deepfake = (fn['p_grayscale'] >= 0.5).sum()

print(f"Anzahl nicht erkannter Deepfakes {len(fn)}")
print(f"Durchschnittliche Anzahl richtiger Modelle {fn['num_of_correct_models'].mean()}")
print(f"Anzahl richtig identifizierter Deepfakes vom Human Modell {p_human_right_for_deepfake}")
print(f"Anzahl richtig identifizierter Deepfakes vom Landscape Modell {p_landscape_right_for_deepfake}")
print(f"Anzahl richtig identifizierter Deepfakes vom Building Modell {p_building_right_for_deepfake}")
print(f"Anzahl richtig identifizierter Deepfakes vom Edges Modell {p_edges_right_for_deepfake}")
print(f"Anzahl richtig identifizierter Deepfakes vom frequency Modell {p_frequency_right_for_deepfake}")
print(f"Anzahl richtig identifizierter Deepfakes vom grayscale Modell {p_grayscale_right_for_deepfake} \n")

as_deepfake_identified_images = df[df['prediction']==1]
fp = as_deepfake_identified_images[as_deepfake_identified_images['label']==0].copy()
fp['num_of_correct_models'] =(fp[p_cols] < 0.5).sum(axis=1)

p_human_right_for_not_deepfake = (fp['p_human'] < 0.5).sum()
p_landscape_right_for_not_deepfake = (fp['p_landscape'] < 0.5).sum()
p_building_right_for_not_deepfake = (fp['p_building'] < 0.5).sum()
p_edges_right_for_not_deepfake = (fp['p_edges'] < 0.5).sum()
p_frequency_right_for_not_deepfake = (fp['p_frequency'] < 0.5).sum()
p_grayscale_right_for_not_deepfake = (fp['p_grayscale'] < 0.5).sum()

print(f"Anzahl falsch identifizierter Deepfakes {len(fp)}")
print(f"Durchschnittliche Anzahl richtiger Modelle {fp['num_of_correct_models'].mean()}")
print(f"Anzahl richtiger identifizierter Real Bilder vom Human Modell {p_human_right_for_not_deepfake}")
print(f"Anzahl richtiger identifizierter Real Bilder vom Landscape Modell {p_landscape_right_for_not_deepfake}")
print(f"Anzahl richtiger identifizierter Real Bilder vom Building Modell {p_building_right_for_not_deepfake}")
print(f"Anzahl richtiger identifizierter Real Bilder vom Edges Modell {p_edges_right_for_not_deepfake}")
print(f"Anzahl richtiger identifizierter Real Bilder vom frequency Modell {p_frequency_right_for_not_deepfake}")
print(f"Anzahl richtiger identifizierter Real Bilder vom grayscale Modell {p_grayscale_right_for_not_deepfake}")