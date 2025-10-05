
# 🩺 Breast Cancer Classification using Decision Trees, Random Forests, AdaBoost, Naive Bayes & PCA

## 📌 Overview

This project presents a comparative machine learning study to classify breast tumors as **benign (0)** or **malignant (1)**. Using a dataset of 569 samples and 30 diagnostic features derived from fine needle aspirate (FNA) images, it explores how various algorithms — Decision Trees, Random Forests, AdaBoost, and Naive Bayes — perform under different configurations, both with and without dimensionality reduction using PCA.

The goal is to demonstrate trade-offs between model interpretability, computational efficiency, and predictive performance in a healthcare classification setting.

---

## 🎯 Objective

Early detection of breast cancer significantly improves survival rates. This project highlights how machine learning techniques can support medical decision-making by identifying patterns in diagnostic data and delivering interpretable, accurate predictions for clinical use.

---

## 📂 Dataset

**File:** `data/Breast_Mass.csv`
**Instances:** 569
**Features:** 30 (mean, standard error, and worst values for radius, texture, perimeter, area, smoothness, etc.)
**Target Variable:**

* `0` → Benign
* `1` → Malignant

The dataset originates from digitized FNA images, each containing real-valued features computed from the cell nuclei.

---

## ⚙️ Workflow

### 1. Data Preparation

* Handled missing values and normalized all numerical features
* Split dataset into training (70%) and testing (30%) sets

### 2. Model Development

**Decision Trees**

* Criteria: Gini Impurity & Entropy
* Depth tuning (1–20), pruning experiments
* Visualization using Graphviz

**Random Forests**

* Estimators tested: 10, 50, 100, 500, 1000
* 5-fold cross-validation
* Feature importance via Mean Decrease in Impurity (MDI) and Permutation Importance

**AdaBoost**

* Estimators tested: 10, 50, 100, 500, 1000
* 5-fold cross-validation for evaluation

**Naive Bayes**

* Gaussian Naive Bayes used as a baseline classifier

**Principal Component Analysis (PCA)**

* Reduced dimensionality while retaining >95% explained variance
* Compared Random Forest performance on reduced vs full feature sets

---

## 🛠️ Tools & Libraries

* **Python 3.10+**
* **Pandas, NumPy** – data handling and numerical processing
* **Scikit-learn** – model training, evaluation, and PCA
* **Matplotlib** – visual analysis and performance plots
* **Graphviz** – tree structure visualization
* **Jupyter Notebook** – interactive analysis

---

## 🚀 How to Run

### Clone the repository

```bash
git clone https://github.com/VenkataSathyaBoppana/breast-cancer-ml-ensemble
cd breast-cancer-ml-ensemble
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Launch the notebook

```bash
jupyter notebook notebooks/BreastMass_Classification.ipynb
```

### Or run scripts directly

```bash
python scripts/DecisionTree.py
python scripts/RandomForest.py
python scripts/AdaBoost.py
python scripts/NaiveBayes.py
python scripts/PCA_RF.py
```

---

## 📁 Repository Structure

```
data/
 └── Breast_Mass.csv
notebooks/
 └── BreastMass_Classification.ipynb
scripts/
 ├── DecisionTree.py
 ├── RandomForest.py
 ├── AdaBoost.py
 ├── NaiveBayes.py
 └── PCA_RF.py
README.md
requirements.txt
```

---

## 🔬 Key Insights

* Decision Trees provided the best interpretability
* Random Forests achieved the highest accuracy and robustness
* AdaBoost improved stability with ensemble learning
* Naive Bayes served as a lightweight, quick baseline
* PCA reduced dimensionality with minimal performance loss

---

## 📊 Visualizations

* Decision Tree diagrams (Entropy vs Gini)
* Accuracy vs tree depth and number of estimators
* Feature importance comparisons (MDI vs Permutation)
* PCA explained variance plot and reduced-space visualization

---

## 🎯 Outcomes

This study underscores how machine learning can support **early breast cancer screening** by balancing transparency and predictive strength.

* **Interpretability**: Decision Trees
* **Robustness**: Random Forests, AdaBoost
* **Simplicity**: Naive Bayes
* **Efficiency**: PCA

---

## 🧠 Citation

If referencing this work, please cite:

> Boppana, V. S. (2025). *Breast Cancer Classification using Decision Trees, Random Forests, AdaBoost, Naive Bayes & PCA.*

---

