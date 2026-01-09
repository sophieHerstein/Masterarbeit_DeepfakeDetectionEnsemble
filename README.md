# Masterarbeit – Codebasis

Dieses Repository enthält den im Rahmen meiner Masterarbeit entwickelten Code zur Umsetzung und Evaluation eines Ensemble-Ansatzes zur Deepfake-Erkennung.

> Hinweis: Dieses Repository dokumentiert die technische Umsetzung der Experimente.
> Theoretische Herleitungen, methodische Entscheidungen sowie die Interpretation der Ergebnisse sind Bestandteil der schriftlichen Masterarbeit und werden hier nicht erneut ausgeführt.

---

## Überblick

Ziel dieses Projekts ist die Untersuchung eines Ensemble-Ansatzes zur Deepfake-Erkennung.
Der Code dient der Erstellung von Fake-Bildern für den eigenen Datensatz, der Vorbereitung der Bilder für Test, Validierung und Training, der Ensemble-Implementierung, der Durchführung von Trainings- und Testläufen, der Auswertung verschiedener Testfälle sowie der Darstellung der Ergebnisse über ein Dashboard.

Nicht enthalten sind:

* Datensätze (Lizenz- und Größenbeschränkungen)
* trainierte Modellgewichte
* generierte Deepfake-Medien

---

## Projektstruktur

```
root/
├── data/                     # Platzhalter für Datensätze (nicht enthalten)
│   ├──  classifier/
│   │   ├── train/
│   │   └── val/
│   ├── test/
│   ├── train/
│   └── val/
│
├── logs/
│   ├── image_manipulation/   # Logs zur Deepfake-Erstellung durch manipulierte Real-Bilder (Prompts, Seeds)
│   ├── image_synthesis/      # Logs zur synthetischen Deepfake-Erstellung (Prompts, Seeds)
│   ├── test/                 # Testergebnisse
│   │   └── ensemble/         # Ensemble-spezifische Testergebnisse
│   └── train/                # Trainingslogs (CSV)
│
├── utils/                    # diverse Hilfsskripts
│   ├── data_preparation/     # für die Datenerstellung und -vorbereitung
│   ├── ensemble/             # fürs Ensemble
│   └── result_dashboard.py   # Streamlit-Dashboard zur Ergebnisanalyse
│
├── classifier.py             # Implementierung des Inhalts-Klassifizierers für die Gewichtung
├── ensemble.py               # Definition des Ensemble-Modells
├── meta-classifier.py        # Training und Tests des Meta-Classifiers
├── test.py                   # Testpipeline mit mehreren Testfällen
├── train.py                  # Trainingslogik
│
├── requirements.txt
├── THIRD_PARTY_LICENSES.md
└── README.md
```

---

## Datensätze

Aus lizenzrechtlichen Gründen sind **keine Datensätze** im Repository enthalten.

Die in der Arbeit verwendeten Datensätze sowie deren Lizenzen sind:

* in der **Masterarbeit** dokumentiert
* zusätzlich in der Datei `THIRD_PARTY_LICENSES.md` aufgeführt

---

## Training

Das Training der einzelnen Detektoren erfolgt über das Skript `train.py`.

```bash
python train.py
```

Dabei werden u. a. folgende Informationen protokolliert:

* Trainings- und Validierungsmetriken
* Epochendauer und Gesamttrainingszeit
* Modellbezogene Logs als CSV-Dateien

Trainierte Modelle werden **nicht** im Repository gespeichert.

---

## Test & Evaluation

Die Evaluation erfolgt zentral über `test.py`.

```bash
python test.py
```

Das Skript enthält mehrere vordefinierte Testfälle, die automatisiert durchlaufen werden.
Erfasst werden unter anderem:

* Accuracy, Precision, Recall, F1-Score
* Ensemble-spezifische Testergebnisse
* Ergebnislogs als CSV-Dateien

Die Ergebnisse werden unter `logs/test/` bzw. `logs/test/ensemble/` abgelegt.

---

## Ergebnisse & Outputs

Die erzeugten Outputs bestehen primär aus:

* CSV-Dateien mit Test- und Trainingsmetriken
* später generierten Plots (z. B. Confusion Matrizen)
* strukturierten Logdateien

Eine grafische Aufbereitung erfolgt **nicht automatisch** im Training/Test, sondern über das Dashboard bzw. mit `utils\get_ensemble_plots.py` und `utils\get_result_plots.py`.

```bash
python utils\get_ensemble_plots.py
```
```bash
python utils\get_result_plots.py
```

---

## Dashboard

Zur Analyse der Trainings- und Testergebnisse steht ein Streamlit-Dashboard zur Verfügung.

Start aus dem Projekt-Root:

```bash
streamlit run utils/result_dashboard.py
```

Funktionen des Dashboards:

* Übersicht über Trainings- und Testergebnisse
* Vergleich verschiedener Testläufe
* Darstellung der Metriken

---

## Deepfake-Erstellung (Logs)

Das Repository enthält zusätzliche Skripte und Logs zur Erstellung von Deepfakes.
Dabei werden u. a. verwendete Prompts und Seeds protokolliert, um die Nachvollziehbarkeit der Generierung zu gewährleisten.

Die erzeugten Medien selbst sind nicht enthalten.

---


## Lizenz & Nutzung

Die verwendeten Drittbibliotheken und Datensätze sind in `THIRD_PARTY_LICENSES.md` dokumentiert.

Der Code dient ausschließlich wissenschaftlichen Zwecken im Rahmen der Masterarbeit.
