# Data Domain: The Iris Benchmark Dataset
**Industrial Training Kit Material** | Project 2 | [cite_start]Batch: 2026 

## 📊 Dataset Profile
The Iris dataset is a classic, multi-class classification benchmark widely utilized to test supervised machine learning workflows. [cite_start]It represents a completely balanced domain layout.

* [cite_start]**Total Sample Target Space:** 150 instances 
* [cite_start]**Distribution Balance:** Exactly 50 samples per class 
* [cite_start]**Target Categories (3 Class Labels):** 1. `Iris-setosa` (Class 0) 
  2. [cite_start]`Iris-versicolor` (Class 1) 
  3. [cite_start]`Iris-virginica` (Class 2) 

---

## 📐 Dimensional Feature Layout
Each sample profile captures four separate baseline physical vector measurements (all recorded in centimeters) that the classification algorithm uses to discover spatial boundaries:

| Feature Index | Feature Coordinate Name | Domain Dimension | Metric |
| :--- | :--- | :--- | :--- |
| **Feature 1** | `sepal length (cm)` | Vertical Sepal Outer Frame  | [cite_start]Quantitative ($cm$)  |
| **Feature 2** | `sepal width (cm)` | Horizontal Sepal Outer Frame | [cite_start]Quantitative ($cm$)  |
| **Feature 3** | `petal length (cm)` | Vertical Petal Inner Frame | [cite_start]Quantitative ($cm$) |
| **Feature 4** | `petal width (cm)` | Horizontal Petal Inner Frame  | Quantitative ($cm$) |

---

## ⚙️ Processing Constraints Applied
To prepare this dataset for the K-Nearest Neighbors (KNN) framework, the following structural rules from the master pipeline are implemented:

1. **Random Shuffle Separation:** The rows are randomized to eliminate sequential sampling capture order biases.
2. **80/20 Partition Matrix:** 120 rows are allocated to the algorithmic pattern recognition matrix (`X_train`), while 30 distinct rows are entirely isolated for final validation diagnostics (`X_test`).
3. **Standard Scale Bounds:** The data vectors are standardized using $StandardScaler$ to map the variables to a unified Mean of $0$ and a Variance of $1$, ensuring magnitude variations do not warp Euclidean distance calculations.
