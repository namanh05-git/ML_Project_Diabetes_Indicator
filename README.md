# Diabetes Health Indicators: Exploration and Classification

A two-stage data science project on the CDC BRFSS 2015 health survey data (253,680
respondents). Stage 1 explores how lifestyle and health factors differ across diabetes status;
Stage 2 trains a decision tree to predict it.

## Contents

| Notebook | What it does |
|---|---|
| `notebooks/Project_Stage1_data_preprocessing.ipynb` | Exploratory analysis. Computes, for each diabetes class (none / prediabetes / diabetes), the proportion of respondents with each risk factor (high blood pressure, high cholesterol, smoking, stroke, heart disease, physical activity, diet, alcohol, healthcare access, mobility). Visualized as a heatmap and one stacked bar chart per factor, plus diabetes status by age group and a BMI boxplot. |
| `notebooks/Project_Stage2_decision_tree_classifier.ipynb` | Classification. Removes BMI outliers (IQR rule, leaving 243,833 rows), bins BMI into WHO categories, one-hot encodes BMI and age, and splits the data 75/10/15 into train/validation/test. Trains a depth-4 `DecisionTreeClassifier` with balanced class weights and evaluates it with accuracy and a confusion matrix. |

## Results (Stage 2)

The three classes are heavily imbalanced (about 85% no diabetes, 2% prediabetes, 13% diabetes),
so class weights are balanced. The depth-4 tree reaches **61.9% validation accuracy**. Accuracy
alone is misleading on imbalanced data, so check the confusion matrix in the notebook.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── data/
│   ├── README.md
│   └── diabetes.csv
└── notebooks/
    ├── Project_Stage1_data_preprocessing.ipynb
    └── Project_Stage2_decision_tree_classifier.ipynb

```

## Running

```bash
git clone <this-repo-url>
cd <repo-name>
pip install -r requirements.txt
cd notebooks
jupyter notebook
```

Run the notebooks from inside `notebooks/` so the relative path `../data/diabetes.csv` resolves.

## Tech stack

Python 3, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.
