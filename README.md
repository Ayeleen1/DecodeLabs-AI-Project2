# Project 2: Data Classification Using AI
**Industrial Training Kit** | Batch: 2026 | Powered by DecodeLabs

## 📋 Project Overview
This project transitions away from traditional manual rule-based heuristics and moves into Supervised Machine Learning. The objective is to build an end-to-end data classification pipeline that trains a machine to recognize distinct patterns in data and automatically categorize new information based on mathematical logic.

---

## 📐 The Master Blueprint: IPO Framework

### 1. INPUT PHASE
* **Dataset:** The Iris Benchmark Dataset consisting of 150 balanced samples.
* **Target Classes (3):** Setosa, Versicolor, and Virginica.
* **Dimensions (4 Features):** Sepal Length, Sepal Width, Petal Length, and Petal Width (measured in cm).

### 2. PROCESS PHASE
* **Train-Test Split:** The dataset is split into an 80% training set (for pattern recognition) and a 20% test set (for validation) to completely insulate testing logic.
* **Shuffling:** Rows are randomized before parsing to prevent systemic data grouping bias.
* **Feature Scaling:** Applied `StandardScaler` to normalize the data vectors to a Mean of 0 and a Variance of 1. This prevents larger feature bounds from overpowering the spatial distance metrics.
* **Algorithmic Model:** Deployed the **K-Nearest Neighbors (KNN)** algorithm ($K=5$) to derive continuous spatial decision boundaries across feature space.

### 3. OUTPUT PHASE (Diagnostics)
The trained model evaluates unseen test patterns and generates validation benchmarks. Instead of relying blindly on raw accuracy, performance is verified through:
* **A Confusion Matrix:** Tracking structural hits and errors (True Positives, True Negatives, Type I False Alarms, and Type II Missed Detections).
* **F1 Score:** Evaluating the harmonic mean balance between Precision and Recall.

---

## 💻 Python Core Implementation

```python
# Save this code in a file named: src/classify_iris.py
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# 1. LOAD DATA
iris = load_iris(as_frame=True)
X, y = iris.data, iris.target

# 2. PARTITION & SCALE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. TRAIN KNN MODEL
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# 4. PREDICT & VALIDATE
predictions = model.predict(X_test_scaled)
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions, target_names=iris.target_names))
