import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
import plotly.graph_objects as go

from config import TRAININGS_VARIANTEN, TEST_VARIANTEN
from config import MODELS

st.set_page_config(page_title="Masterarbeit Deepfake Detection Ensemble Model Dashboard", layout="wide")


# === Helper Functions ===

@st.cache_data
def load_summary(train_type):
    try:
        return pd.read_csv(f"logs/train/{train_type}/train_results.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=["Modell", "Train-Acc", "Val-Acc", "Loss", "Trainzeit (s)", "last epoch"])


def load_training_log(model, train_type):
    path = f"logs/train/{train_type}/{model}.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_test_results(model):
    path = f"logs/test/{model}_metrics.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def load_ensemble_results(model, test_dir):
    path = f"logs/test/ensemble/{model}_{test_dir}_details.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def plot_line_chart(df, x, y, title):
    fig, ax = plt.subplots()
    ax.plot(df[x], df[y])
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    st.pyplot(fig)


def plot_confusionmatrix(df):
    cm = np.array(df)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

def plot_grouped_bar_chart(delta_df, title):
    if delta_df is None or delta_df.empty:
        return

    fig = go.Figure()
    for metric in delta_df.columns:
        fig.add_trace(go.Bar(
            x=delta_df.index,
            y=delta_df[metric],
            name=metric
        ))

    fig.update_layout(
        barmode='group',
        title=title,
        xaxis_title="Variante",
        yaxis_title="Δ-Wert",
        legend_title="Metrik",
        template="plotly_white",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

def compute_deltas(model):
    variations = ["jpeg", "noisy", "scaled"]
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

    df = load_test_results(model)
    if df is None or df.empty:
        return None, None

    df_model = df[df["Modell"] == model]

    # Basiszeilen (Standardtests)
    base_known = df_model[df_model["TestVariante"] == "known_test_dir"]
    base_unknown = df_model[df_model["TestVariante"] == "unknown_test_dir"]

    if base_known.empty or base_unknown.empty:
        return None, None

    base_known = base_known.iloc[0]
    base_unknown = base_unknown.iloc[0]

    # Dictionaries für Ergebnisse
    known_deltas = {}
    unknown_deltas = {}

    # Schleife über Varianten
    for var in variations:
        # Known
        var_known = df_model[df_model["TestVariante"] == f"known_test_{var}_dir"]
        if not var_known.empty:
            var_known = var_known.iloc[0]
            known_deltas[var] = {m: round(var_known[m] - base_known[m], 4) for m in metrics}

        # Unknown
        var_unknown = df_model[df_model["TestVariante"] == f"unknown_test_{var}_dir"]
        if not var_unknown.empty:
            var_unknown = var_unknown.iloc[0]
            unknown_deltas[var] = {m: round(var_unknown[m] - base_unknown[m], 4) for m in metrics}

    # In DataFrames umwandeln
    known_df = pd.DataFrame(known_deltas).T if known_deltas else None
    unknown_df = pd.DataFrame(unknown_deltas).T if unknown_deltas else None

    return known_df, unknown_df

# === Sidebar Selection ===
st.sidebar.title("Modellauswahl")

models = MODELS
model = st.sidebar.selectbox("Modell", [*models, "weighted_ensemble", "unweighted_ensemble", "unweighted_meta_classifier_ensemble", "weighted_meta_classifier_ensemble"]) if models else None

train_types = TRAININGS_VARIANTEN
test_types = TEST_VARIANTEN

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Übersicht", "📈 Training", "🧪 Testmetriken", "💪🏼 Robustheit", "🏆 Vergleich der Testergebnisse", "🔍 Ensemble Analyse"
])
# === Übersicht ===
with tab1:
    st.header("📋 Modellübersicht")
    # === Load Data ===
    if model:
        st.subheader("Trainingszusammenfassung")
        for training_type in train_types:
            summary_df = load_summary(training_type)
            row = summary_df[(summary_df["Modell"] == model)]
            st.subheader(training_type)
            st.dataframe(row, hide_index=True, use_container_width=True)
    else:
        st.info("Bitte wähle Modell und Trainingsdaten aus der Seitenleiste.")

# === Training ===
with tab2:
    st.header("📈 Trainingsverlauf")
    training_type = st.selectbox("Trainingstyp", train_types, key="trainingsdaten_tab2") if train_types else None
    if model and training_type:
        log_df = load_training_log(model, training_type)
        if log_df is not None:
            plot_line_chart(log_df, "Epoche", "Train-Acc", "Train Accuracy")
            plot_line_chart(log_df, "Epoche", "Val-Acc", "Validation Accuracy")
            plot_line_chart(log_df, "Epoche", "Loss", "Loss")
        else:
            st.warning("Keine Trainingsdaten gefunden.")
    else:
        st.info("Bitte wähle Modell und Trainingsdaten aus.")

# === Testmetriken ===
with tab3:
    st.header("🧪 Testmetriken")
    test_type = st.selectbox("Testarten", test_types, key="testtypes_tab3") if test_types else None
    if model:
        test_df = load_test_results(model)
        if test_df is not None:
            df = test_df.loc[test_df['TestVariante'] == test_type]
            st.dataframe(df, hide_index=True)
            metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score"]
            if all(col in df.columns for col in metric_cols):
                plot_confusionmatrix([[df["TN"].iloc[0], df["FP"].iloc[0]],
                                      [df["FN"].iloc[0], df["TP"].iloc[0]]])
                fig, ax = plt.subplots()
                df[metric_cols].T.plot(kind="bar", ax=ax, legend=False)
                ax.set_title(f"{model})")
                ax.set_ylabel("Wert")
                plt.xticks(rotation=45)
                st.pyplot(fig)
        else:
            st.warning("Keine Testdaten gefunden.")
    else:
        st.info("Bitte wähle ein Modell aus.")

# === Robustheit ===
with tab4:
    st.header("🧪 Robustheitsvergleich (Δ Metriken vs. Standard)")
    if model:
        known_df, unknown_df = compute_deltas(model)

        if known_df is not None:
            st.subheader("📘 Known-Test Δ-Werte")
            st.dataframe(known_df, use_container_width=True)
            plot_grouped_bar_chart(known_df, f"Δ-Metriken – Known – {model}")
        else:
            st.warning("Keine Daten für known-Test gefunden.")

        if unknown_df is not None:
            st.subheader("📙 Unknown-Test Δ-Werte")
            st.dataframe(unknown_df, use_container_width=True)
            plot_grouped_bar_chart(unknown_df, f"Δ-Metriken – Unknown – {model}")
        else:
            st.warning("Keine Daten für unknown-Test gefunden.")
    else:
        st.info("Bitte wähle ein Modell aus.")

# === Vergleich der Testergebnisse ===
with tab5:
    st.header("🏆 Testergebnisse")
    test_type = st.selectbox("Testarten", test_types, key="testtypes_tab5") if test_types else None

    # === Daten sammeln ===
    data = []
    for m in [*MODELS, "weighted_ensemble", "unweighted_ensemble", "unweighted_meta_classifier_ensemble", "weighted_meta_classifier_ensemble"]:
        test_df = load_test_results(m)
        df = test_df.loc[test_df['TestVariante'] == test_type]
        if df.empty:
            continue
        data.append({
            "Model": m,
            "Accuracy": df["Accuracy"].iloc[0],
            "Precision": df["Precision"].iloc[0],
            "Recall": df["Recall"].iloc[0],
            "F1-Score": df["F1-Score"].iloc[0],
            "ROC-AUC": df["ROC-AUC"].iloc[0]
        })

    if not data:
        st.warning("Keine Daten für diese Testvariante gefunden.")
        st.stop()

    # === Sortier-Option ===
    sort_metric = st.radio(
        "Sortiere nach:",
        ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        horizontal=True,
        key="sort_metric_tab5"
    )

    # === Sortieren ===
    data = sorted(data, key=lambda x: x[sort_metric], reverse=True)

    # === Plot vorbereiten ===
    models = [d["Model"] for d in data]
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

    ax.set_title(f"Modellvergleich ({test_type}) – sortiert nach {sort_metric}")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    plt.tight_layout()

    st.pyplot(fig)

with tab6:
    st.header("🔍 Ensemble Analyse")
    if model in ["weighted_ensemble", "unweighted_ensemble", "unweighted_meta_classifier_ensemble", "weighted_meta_classifier_ensemble"]:
        test_type = st.selectbox("Testarten", test_types, key="testtypes_tab7") if test_types else None

        df = load_ensemble_results(model, test_type)
        p_cols = ['p_human', 'p_landscape', 'p_building', 'p_edges', 'p_frequency', 'p_grayscale']
        w_cols = ['w_human', 'w_landscape', 'w_building', 'w_edges', 'w_frequency', 'w_grayscale']

        st.subheader("Nicht als Deepfake identifizierte Bilder - False Negatives")
        not_as_deepfake_identified_images = df[df['prediction'] == 0]
        fn = not_as_deepfake_identified_images[not_as_deepfake_identified_images['label'] == 1].copy()
        st.dataframe(fn, hide_index=True, use_container_width=True)
        fn['num_of_correct_models'] = (fn[p_cols] >= 0.5).sum(axis=1)

        p_human_right_for_deepfake = (fn['p_human'] >= 0.5).sum()
        p_landscape_right_for_deepfake = (fn['p_landscape'] >= 0.5).sum()
        p_building_right_for_deepfake = (fn['p_building'] >= 0.5).sum()
        p_edges_right_for_deepfake = (fn['p_edges'] >= 0.5).sum()
        p_frequency_right_for_deepfake = (fn['p_frequency'] >= 0.5).sum()
        p_grayscale_right_for_deepfake = (fn['p_grayscale'] >= 0.5).sum()

        st.subheader("Statistiken")
        st.text(f"Anzahl nicht erkannter Deepfakes {len(fn)}")
        st.text(f"Min. Prob {fn['final_prob'].min()}")
        st.text(f"Max. Prob {fn['final_prob'].max()}")
        st.text(f"Mean Prob {fn['final_prob'].mean()}")
        st.text(f"Durchschnittliche Anzahl richtiger Modelle {fn['num_of_correct_models'].mean()}")
        st.text(f"Anzahl richtig identifizierter Deepfakes vom Human Modell {p_human_right_for_deepfake}")
        st.text(f"Anzahl richtig identifizierter Deepfakes vom Landscape Modell {p_landscape_right_for_deepfake}")
        st.text(f"Anzahl richtig identifizierter Deepfakes vom Building Modell {p_building_right_for_deepfake}")
        st.text(f"Anzahl richtig identifizierter Deepfakes vom Edges Modell {p_edges_right_for_deepfake}")
        st.text(f"Anzahl richtig identifizierter Deepfakes vom frequency Modell {p_frequency_right_for_deepfake}")
        st.text(f"Anzahl richtig identifizierter Deepfakes vom grayscale Modell {p_grayscale_right_for_deepfake} \n")

        st.subheader("Nicht als Real Bilder identifizierte Bilder - False Positives")
        as_deepfake_identified_images = df[df['prediction'] == 1]
        fp = as_deepfake_identified_images[as_deepfake_identified_images['label'] == 0].copy()
        st.dataframe(fp, hide_index=True, use_container_width=True)
        fp['num_of_correct_models'] = (fp[p_cols] < 0.5).sum(axis=1)

        p_human_right_for_not_deepfake = (fp['p_human'] < 0.5).sum()
        p_landscape_right_for_not_deepfake = (fp['p_landscape'] < 0.5).sum()
        p_building_right_for_not_deepfake = (fp['p_building'] < 0.5).sum()
        p_edges_right_for_not_deepfake = (fp['p_edges'] < 0.5).sum()
        p_frequency_right_for_not_deepfake = (fp['p_frequency'] < 0.5).sum()
        p_grayscale_right_for_not_deepfake = (fp['p_grayscale'] < 0.5).sum()

        st.subheader("Statistiken")
        st.text(f"Anzahl falsch identifizierter Deepfakes {len(fp)}")
        st.text(f"Min. Prob {fp['final_prob'].min()}")
        st.text(f"Max. Prob {fp['final_prob'].max()}")
        st.text(f"Mean Prob {fp['final_prob'].mean()}")
        st.text(f"Durchschnittliche Anzahl richtiger Modelle {fp['num_of_correct_models'].mean()}")
        st.text(f"Anzahl richtiger identifizierter Real Bilder vom Human Modell {p_human_right_for_not_deepfake}")
        st.text(f"Anzahl richtiger identifizierter Real Bilder vom Landscape Modell {p_landscape_right_for_not_deepfake}")
        st.text(f"Anzahl richtiger identifizierter Real Bilder vom Building Modell {p_building_right_for_not_deepfake}")
        st.text(f"Anzahl richtiger identifizierter Real Bilder vom Edges Modell {p_edges_right_for_not_deepfake}")
        st.text(f"Anzahl richtiger identifizierter Real Bilder vom frequency Modell {p_frequency_right_for_not_deepfake}")
        st.text(f"Anzahl richtiger identifizierter Real Bilder vom grayscale Modell {p_grayscale_right_for_not_deepfake}")
    else:
        st.info("Bitte wähle ein Ensemble Modell aus der Seitenleiste")
