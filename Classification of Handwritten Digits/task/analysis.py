import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer


def load_and_preprocess_data(limit=6000, test_size=0.3, random_state=40):
    """
    Step 1: Load MNIST data, flatten, subset, split, and normalize.
    """
    # 1. Load Data
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

    # 2. Reshape (Flatten 28x28 images to 784 vector)
    n, h, w = x_train.shape
    x_flattened = x_train.reshape(n, h * w)

    # 3. Create Subset (for faster processing)
    x_subset = x_flattened[:limit]
    y_subset = y_train[:limit]

    # 4. Train-Test Split
    x_train, x_test, y_train, y_test = train_test_split(
        x_subset, y_subset, test_size=test_size, random_state=random_state
    )

    # 5. Normalization (Scaling values to 0-1 range)
    # Note: fit_transform on TRAIN, transform only on TEST to prevent data leakage
    normalizer = Normalizer()
    x_train_norm = normalizer.fit_transform(x_train)
    x_test_norm = normalizer.transform(x_test)

    print(f"Data Loaded: {x_train_norm.shape[0]} training samples, {x_test_norm.shape[0]} test samples.")
    return x_train_norm, x_test_norm, y_train, y_test


def evaluate_baseline_models(x_train, x_test, y_train, y_test):
    """
    Step 2: Train and evaluate baseline models with default parameters.
    Returns the top 2 performing models to be tuned.
    """
    print("\n--- Evaluating Baseline Models ---")

    # Initialize models with necessary default overrides
    models = [
        KNeighborsClassifier(),
        DecisionTreeClassifier(random_state=40),
        LogisticRegression(random_state=40, max_iter=10000),  # Increased iter for convergence
        RandomForestClassifier(random_state=40)
    ]

    scores = []

    for model in models:
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        acc = accuracy_score(y_test, preds)

        # Save score and model object
        scores.append({'model': model, 'name': type(model).__name__, 'accuracy': acc})
        print(f"Model: {type(model).__name__} | Accuracy: {acc:.3f}")

    # Sort by accuracy descending
    sorted_scores = sorted(scores, key=lambda x: x['accuracy'], reverse=True)

    # Return top 2 model objects for tuning
    print(f"\nTop 2 Models selected for tuning: {sorted_scores[0]['name']} & {sorted_scores[1]['name']}")
    return sorted_scores[0]['model'], sorted_scores[1]['model']


def tune_hyperparameters(model_1, model_2, x_train, y_train):
    """
    Step 3: Use GridSearchCV to find the best hyperparameters for the top models.
    """
    print("\n--- Starting Grid Search (This may take time...) ---")

    # Define Parameter Grids
    # Note: These are slightly expanded for better results
    param_grid_knn = {
        'n_neighbors': [3, 4, 5, 6],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'brute'],
        'p': [1, 2]  # 1=Manhattan, 2=Euclidean
    }

    param_grid_rf = {
        'n_estimators': [300, 500, 700],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', 'balanced_subsample'],
        'bootstrap': [True, False]
    }

    # Setup Grid Search
    # Note: Using n_jobs=-1 to use all CPU cores
    grid_knn = GridSearchCV(estimator=model_1, param_grid=param_grid_knn, scoring='accuracy', cv=5, n_jobs=-1,
                            verbose=1)
    grid_rf = GridSearchCV(estimator=model_2, param_grid=param_grid_rf, scoring='accuracy', cv=5, n_jobs=-1, verbose=1)

    # Train
    print("Tuning K-Nearest Neighbors...")
    grid_knn.fit(x_train, y_train)

    print("Tuning Random Forest...")
    grid_rf.fit(x_train, y_train)

    return grid_knn, grid_rf


def final_evaluation(grid_knn, grid_rf, x_test, y_test):
    """
    Step 4: Evaluate the optimized models on the held-out test set.
    """
    print("\n--- Final Evaluation ---")

    # KNN Evaluation
    best_knn = grid_knn.best_estimator_
    knn_pred = best_knn.predict(x_test)
    knn_acc = accuracy_score(y_test, knn_pred)

    print(f"K-Nearest Neighbors Algorithm")
    print(f"Best Estimator: {best_knn}")
    print(f"Accuracy: {knn_acc:.3f}\n")

    # Random Forest Evaluation
    best_rf = grid_rf.best_estimator_
    rf_pred = best_rf.predict(x_test)
    rf_acc = accuracy_score(y_test, rf_pred)

    print(f"Random Forest Algorithm")
    print(f"Best Estimator: {best_rf}")
    print(f"Accuracy: {rf_acc:.3f}")


# --- Main Execution Block ---
if __name__ == "__main__":
    # 1. Pipeline: Load Data
    x_train, x_test, y_train, y_test = load_and_preprocess_data()

    # 2. Pipeline: Baseline Check (Optional, but good for comparison)
    # In this script, we hardcode KNN and RF for the tuning step based on project history,
    # but normally we would use the return values from this function.
    evaluate_baseline_models(x_train, x_test, y_train, y_test)

    # 3. Pipeline: Hyperparameter Tuning
    # We create fresh instances for the grid search
    knn_base = KNeighborsClassifier()
    rf_base = RandomForestClassifier(random_state=40)

    grid_knn_result, grid_rf_result = tune_hyperparameters(knn_base, rf_base, x_train, y_train)

    # 4. Pipeline: Final Results
    final_evaluation(grid_knn_result, grid_rf_result, x_test, y_test)