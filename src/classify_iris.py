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
