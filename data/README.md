# Data Domain: The Iris Benchmark Dataset
**Industrial Training Kit Material** | Project 2 | [cite_start]Batch: 2026 [cite: 3, 4]

## 📊 Dataset Profile
[cite_start]The Iris dataset is a classic, multi-class classification benchmark widely utilized to test supervised machine learning workflows[cite: 7, 87]. [cite_start]It represents a completely balanced domain layout[cite: 92].

* [cite_start]**Total Sample Target Space:** 150 instances [cite: 91, 92]
* [cite_start]**Distribution Balance:** Exactly 50 samples per class [cite: 92]
* [cite_start]**Target Categories (3 Class Labels):** 1. `Iris-setosa` (Class 0) [cite: 40, 88]
  2. [cite_start]`Iris-versicolor` (Class 1) [cite: 40, 89]
  3. [cite_start]`Iris-virginica` (Class 2) [cite: 40, 90]

---

## 📐 Dimensional Feature Layout
[cite_start]Each sample profile captures four separate baseline physical vector measurements (all recorded in centimeters) that the classification algorithm uses to discover spatial boundaries[cite: 27, 73, 95]:

| Feature Index | Feature Coordinate Name | Domain Dimension | Metric |
| :--- | :--- | :--- | :--- |
| **Feature 1** | `sepal length (cm)` | [cite_start]Vertical Sepal Outer Frame [cite: 35] | [cite_start]Quantitative ($cm$) [cite: 96] |
| **Feature 2** | `sepal width (cm)` | Horizontal Sepal Outer Frame | [cite_start]Quantitative ($cm$) [cite: 101] |
| **Feature 3** | `petal length (cm)` | Vertical Petal Inner Frame | [cite_start]Quantitative ($cm$) [cite: 98] |
| **Feature 4** | `petal width (cm)` | [cite_start]Horizontal Petal Inner Frame [cite: 39] | [cite_start]Quantitative ($cm$) [cite: 102] |

---

## ⚙️ Processing Constraints Applied
[cite_start]To prepare this dataset for the K-Nearest Neighbors (KNN) framework, the following structural rules from the master pipeline are implemented[cite: 76, 84]:

1. [cite_start]**Random Shuffle Separation:** The rows are randomized to eliminate sequential sampling capture order biases[cite: 141].
2. [cite_start]**80/20 Partition Matrix:** 120 rows are allocated to the algorithmic pattern recognition matrix (`X_train`), while 30 distinct rows are entirely isolated for final validation diagnostics (`X_test`)[cite: 138, 139, 197, 198].
3. [cite_start]**Standard Scale Bounds:** The data vectors are standardized using $StandardScaler$ to map the variables to a unified Mean of $0$ and a Variance of $1$, ensuring magnitude variations do not warp Euclidean distance calculations[cite: 107, 134].
