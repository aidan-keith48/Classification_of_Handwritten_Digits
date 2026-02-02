# Handwritten Digit Classification (MNIST) Pipeline

A comprehensive machine learning system built with **TensorFlow** and **Scikit-Learn** that classifies handwritten digits from the MNIST dataset. The project demonstrates a full ML lifecycle: data preprocessing, baseline benchmarking, and automated hyperparameter tuning.

## 🚀 Project Overview

The goal was to build a system capable of recognizing digits (0-9) from  grayscale images. By transforming spatial pixel data into flat feature vectors and applying advanced normalization techniques, the pipeline achieves high accuracy using ensemble learning and nearest-neighbor algorithms.

## 🛠️ Key Features

* **Tensor Manipulation:** Flattened 3D image arrays into 2D feature matrices suitable for traditional machine learning classifiers.
* **Data Normalization:** Applied `sklearn.preprocessing.Normalizer` to scale pixel intensities, significantly improving convergence for distance-based models.
* **Model Benchmarking:** Evaluated four distinct algorithms:
* K-Nearest Neighbors (KNN)
* Decision Tree
* Logistic Regression
* Random Forest


* **Hyperparameter Optimization:** Utilized `GridSearchCV` with **Cross-Validation** and **Parallel Processing** (`n_jobs=-1`) to find the mathematical "sweet spot" for the best-performing models.

## 📂 Pipeline Stages

1. **ETL & Preprocessing:** Load raw Keras tensors, slice data for efficient training, and normalize features.
2. **Baseline Training:** Establish "out-of-the-box" performance metrics for multiple classifiers.
3. **Grid Search Tuning:** Exhaustively search parameter spaces for `n_neighbors`, `n_estimators`, and `class_weight` to maximize accuracy.
4. **Final Evaluation:** Validated the optimized "Best Estimators" against a held-out test set to ensure model generalizability.

## 💻 Technical Stack

* **Deep Learning:** TensorFlow/Keras (Data Sourcing)
* **Machine Learning:** Scikit-Learn (Pipelines, KNN, Random Forest, GridSearch)
* **Data Science:** NumPy, Pandas
* **Hardware Optimization:** Parallelized training across all available CPU cores.

## 📊 Results & Performance

* **Baseline Winner:** Random Forest (~93.9% accuracy).
* **Optimized KNN:** Achieved significant gains after row-wise normalization.
* **Final Model:** The tuned Random Forest model provides a robust, reproducible solution for handwritten digit recognition.

---

### How to Run

1. Clone the repository.
2. Install dependencies:
```bash
pip install tensorflow scikit-learn numpy pandas

```


3. Run the analysis:
```bash
python analysis.py
