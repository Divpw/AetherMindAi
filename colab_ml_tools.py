# colab_ml_tools.py
# Ensure scikit-learn, numpy, and pandas are installed in your Colab environment.
# You can run this in a Colab cell before importing the module:
# !pip install scikit-learn numpy pandas

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_blobs, make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

def run_ml_task(task_name: str, data_params: dict = None) -> str:
    """
    Runs a specified machine learning task using synthetic data.

    Args:
        task_name (str): The name of the ML task to run.
                         Supported: "classification", "clustering", "regression".
        data_params (dict, optional): Parameters for data generation. Defaults to None.
                                      If None, default parameters will be used.
                                      Example for classification: {'n_samples': 100, 'n_features': 5}

    Returns:
        str: A summary of the result or an error message.
    """
    if data_params is None:
        data_params = {}

    if task_name == "classification":
        # Generate synthetic classification data
        n_samples = data_params.get('n_samples', 100)
        n_features = data_params.get('n_features', 20)
        n_informative = data_params.get('n_informative', 2)
        n_classes = data_params.get('n_classes', 2)
        random_state = data_params.get('random_state', 42)

        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_classes=n_classes,
            random_state=random_state
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

        # Train a Logistic Regression model
        # Could also use DecisionTreeClassifier for variety, as requested
        # from sklearn.tree import DecisionTreeClassifier
        # model = DecisionTreeClassifier(random_state=random_state)
        model = LogisticRegression(random_state=random_state, solver='liblinear')
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        return f"Classification (Logistic Regression) Accuracy: {acc:.4f}"

    elif task_name == "clustering":
        # Generate synthetic clustering data
        n_samples = data_params.get('n_samples', 150)
        n_features = data_params.get('n_features', 2)
        centers = data_params.get('centers', 3)
        cluster_std = data_params.get('cluster_std', 1.0)
        random_state = data_params.get('random_state', 42)

        X, _ = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=centers,
            cluster_std=cluster_std,
            random_state=random_state
        )

        # Train a KMeans model
        model = KMeans(n_clusters=centers, random_state=random_state, n_init='auto')
        model.fit(X)

        return f"Clustering (KMeans) Result: Cluster centers found at:\n{model.cluster_centers_}"

    elif task_name == "regression":
        # Generate synthetic regression data
        n_samples = data_params.get('n_samples', 100)
        n_features = data_params.get('n_features', 1)
        n_informative = data_params.get('n_informative', 1)
        noise = data_params.get('noise', 0.1)
        random_state = data_params.get('random_state', 42)

        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            noise=noise,
            random_state=random_state
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

        # Train a Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # R-squared score could also be a good summary
        # score = model.score(X_test, y_test)
        # return f"Regression (Linear Regression) R-squared: {score:.4f}\nCoefficients: {model.coef_}"
        return f"Regression (Linear Regression) Coefficients: {model.coef_}"

    else:
        return f"Unsupported task: '{task_name}'. Supported tasks are 'classification', 'clustering', 'regression'."

if __name__ == "__main__":
    print("Running ML Task Examples...\n")

    # Example 1: Classification
    print("--- Classification Example ---")
    # Using default data_params for classification
    class_result = run_ml_task("classification")
    print(class_result)
    # Example with custom parameters
    class_params = {'n_samples': 200, 'n_features': 5, 'n_classes': 3, 'random_state': 123}
    class_result_custom = run_ml_task("classification", data_params=class_params)
    print(f"\nCustom Params Classification Result:\n{class_result_custom}\n")

    # Example 2: Clustering
    print("--- Clustering Example ---")
    # Using default data_params for clustering
    cluster_result = run_ml_task("clustering")
    print(cluster_result)
    # Example with custom parameters
    cluster_params = {'n_samples': 200, 'centers': 4, 'cluster_std': 0.8, 'random_state': 123}
    cluster_result_custom = run_ml_task("clustering", data_params=cluster_params)
    print(f"\nCustom Params Clustering Result:\n{cluster_result_custom}\n")

    # Example 3: Regression
    print("--- Regression Example ---")
    # Using default data_params for regression
    reg_result = run_ml_task("regression")
    print(reg_result)
    # Example with custom parameters
    reg_params = {'n_samples': 150, 'n_features': 2, 'noise': 0.5, 'random_state': 123}
    reg_result_custom = run_ml_task("regression", data_params=reg_params)
    print(f"\nCustom Params Regression Result:\n{reg_result_custom}\n")

    # Example 4: Unsupported Task
    print("--- Unsupported Task Example ---")
    unsupported_result = run_ml_task("recommendation")
    print(unsupported_result)

    print("\nNote: To use this module in Colab, first run the following in a cell:")
    print("# !pip install scikit-learn numpy pandas")
    print("# Then, upload this .py file or copy its content into a cell, then import/use.")
