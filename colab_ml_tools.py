# colab_ml_tools.py
# Enhanced module for common ML tasks in Colab.
# Ensure scikit-learn, numpy, pandas, and matplotlib are installed.
# In a Colab cell: !pip install scikit-learn numpy pandas matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification, make_blobs, make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler

# --- Helper Functions ---

def _generate_synthetic_data(task_name: str, data_params: dict):
    """
    Generates synthetic data for specified ML task.

    Args:
        task_name (str): "classification", "regression", or "clustering".
        data_params (dict): Parameters for data generation.
            Required keys:
                'n_samples' (int): Number of samples.
                'random_state' (int): Random seed for reproducibility.
            Optional keys for all:
                'n_features' (int): Number of features (default 2 for easy plotting).
            For "classification":
                'n_classes' (int): Number of classes (default 2).
                'n_informative' (int): Number of informative features (default 2).
            For "regression":
                'n_features' (int): Default 1 for easy plotting.
                'n_informative' (int): Number of informative features (default 1).
                'noise' (float): Noise level (default 0.1).
            For "clustering":
                'n_features' (int): Default 2 for easy plotting.
                'centers' (int): Number of clusters (default 3).
                'cluster_std' (float): Standard deviation of clusters (default 1.0).

    Returns:
        tuple: (X, y) for classification/regression, or (X, None) for clustering.
               Returns (None, None) if task_name is invalid or params are missing.
    """
    required_keys = ['n_samples', 'random_state']
    if not all(key in data_params for key in required_keys):
        print(f"Error: Missing required data_params: {', '.join(required_keys)}")
        return None, None

    n_samples = data_params['n_samples']
    random_state = data_params['random_state']

    X, y = None, None

    if task_name == "classification":
        n_features = data_params.get('n_features', 2)
        n_classes = data_params.get('n_classes', 2)
        n_informative = data_params.get('n_informative', n_features if n_features <= n_classes else n_classes) # Ensure n_informative <= n_features
        if n_informative > n_features: # sklearn constraint
             n_informative = n_features
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_features - n_informative,
            n_classes=n_classes,
            random_state=random_state
        )
    elif task_name == "regression":
        n_features = data_params.get('n_features', 1)
        n_informative = data_params.get('n_informative', n_features)
        if n_informative > n_features: # sklearn constraint
             n_informative = n_features
        noise = data_params.get('noise', 10.0) # Increased default noise for better viz
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            noise=noise,
            random_state=random_state
        )
    elif task_name == "clustering":
        n_features = data_params.get('n_features', 2)
        centers = data_params.get('centers', 3)
        cluster_std = data_params.get('cluster_std', 1.0)
        X, y = make_blobs(  # y here is the true cluster identifier, we return None for y in final tuple
            n_samples=n_samples,
            n_features=n_features,
            centers=centers,
            cluster_std=cluster_std,
            random_state=random_state
        )
        return X, None # For clustering, y is not used in the same way by run_ml_task
    else:
        print(f"Error: Unknown task_name '{task_name}' in _generate_synthetic_data.")
        return None, None

    # Standardize features for better performance of some models
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y


def _train_model(task_name: str, X_train: np.ndarray, y_train: np.ndarray = None, model_params: dict = None):
    """
    Initializes and trains a model for the specified task.

    Args:
        task_name (str): "classification", "regression", or "clustering".
        X_train (np.ndarray): Training data features.
        y_train (np.ndarray, optional): Training data labels. Required for classification/regression.
        model_params (dict, optional): Parameters for model initialization.

    Returns:
        sklearn model: Trained model, or None if task is invalid.
    """
    if model_params is None:
        model_params = {}

    model = None
    if task_name == "classification":
        # Ensure random_state is passed for reproducibility if provided
        rs = model_params.get('random_state', None)
        solver = model_params.get('solver', 'liblinear') # Good default for small datasets
        model = LogisticRegression(random_state=rs, solver=solver, **{k:v for k,v in model_params.items() if k not in ['random_state', 'solver']})
        model.fit(X_train, y_train)
    elif task_name == "regression":
        model = LinearRegression(**model_params)
        model.fit(X_train, y_train)
    elif task_name == "clustering":
        rs = model_params.get('random_state', None)
        n_clusters = model_params.get('n_clusters', 3) # Should match 'centers' from data_params
        # Use 'n_init' to suppress warning, common practice
        model = KMeans(n_clusters=n_clusters, random_state=rs, n_init='auto', **{k:v for k,v in model_params.items() if k not in ['random_state', 'n_clusters', 'n_init']})
        model.fit(X_train) # X_train here is the full dataset for clustering
    else:
        print(f"Error: Unknown task_name '{task_name}' in _train_model.")
        return None
    return model


def _evaluate_model(task_name: str, model, X_eval: np.ndarray, y_eval: np.ndarray = None):
    """
    Evaluates the trained model.

    Args:
        task_name (str): "classification", "regression", or "clustering".
        model: Trained sklearn model.
        X_eval (np.ndarray): Data to evaluate on. For clustering, this is the full dataset.
        y_eval (np.ndarray, optional): True labels for evaluation. Required for classification/regression.

    Returns:
        dict: Dictionary of metrics, or empty dict if task is invalid.
              "classification": {"accuracy": float}
              "regression": {"r2_score": float, "coefficients": np.array}
              "clustering": {"inertia": float, "cluster_centers": np.array}
    """
    metrics = {}
    if task_name == "classification":
        if y_eval is None:
            print("Error: y_eval cannot be None for classification evaluation.")
            return metrics
        predictions = model.predict(X_eval)
        metrics["accuracy"] = accuracy_score(y_eval, predictions)
    elif task_name == "regression":
        if y_eval is None:
            print("Error: y_eval cannot be None for regression evaluation.")
            return metrics
        # predictions = model.predict(X_eval) # Not strictly needed for R^2 with model.score
        metrics["r2_score"] = model.score(X_eval, y_eval) # R2 score
        metrics["coefficients"] = model.coef_
    elif task_name == "clustering":
        metrics["inertia"] = model.inertia_
        metrics["cluster_centers"] = model.cluster_centers_
    else:
        print(f"Error: Unknown task_name '{task_name}' in _evaluate_model.")
    return metrics


def _plot_results(task_name: str, model, X: np.ndarray, y_or_labels: np.ndarray = None, title_suffix: str = ""):
    """
    Generates and displays a plot for the task results.

    Args:
        task_name (str): "classification", "regression", or "clustering".
        model: Trained sklearn model.
        X (np.ndarray): Data features (preferably 2D for classification/clustering, 1D for regression).
        y_or_labels (np.ndarray, optional): True labels or predicted cluster labels.
        title_suffix (str, optional): Suffix for the plot title.
    """
    plt.figure(figsize=(8, 6))
    title = f"{task_name.capitalize()} Results {title_suffix}"

    if task_name == "classification":
        if X.shape[1] != 2:
            print("Plotting for classification is only supported for 2D features.")
            return
        if y_or_labels is None:
            print("Error: y_or_labels required for classification plot.")
            return
        plt.scatter(X[:, 0], X[:, 1], c=y_or_labels, cmap=plt.cm.Paired, edgecolors='k')
        # Could attempt to plot decision boundary if model is LogisticRegression and X is 2D
        # This is more involved, skipping for now to keep it simpler.

    elif task_name == "regression":
        if X.shape[1] != 1:
            print("Plotting for regression is only supported for 1D features.")
            return
        if y_or_labels is None:
            print("Error: y_or_labels required for regression plot.")
            return
        plt.scatter(X, y_or_labels, color='blue', label='Actual data', edgecolors='k')
        plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression line')
        plt.legend()

    elif task_name == "clustering":
        if X.shape[1] != 2:
            print("Plotting for clustering is only supported for 2D features.")
            return
        labels = model.labels_
        centers = model.cluster_centers_
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolors='k', s=50, alpha=0.7)
        plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.9, marker='X', label='Centroids')
        plt.legend()

    else:
        print(f"Plotting not supported for task: {task_name}")
        return

    plt.title(title)
    plt.xlabel("Feature 1" if X.shape[1] > 0 else "Feature")
    if X.shape[1] > 1:
        plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()

# --- Core Function ---

def run_ml_task(task_name: str, data_params: dict, model_params: dict = None, plot: bool = False) -> str:
    """
    Runs a specified machine learning task, including data generation,
    model training, evaluation, and optional plotting.

    Args:
        task_name (str): The ML task: "classification", "regression", "clustering".
        data_params (dict): Parameters for synthetic data generation.
                            See `_generate_synthetic_data` docstring for details.
        model_params (dict, optional): Parameters for model initialization.
                                       For "clustering", `n_clusters` here should ideally match
                                       `centers` in `data_params`. If not provided, defaults used.
        plot (bool, optional): If True, generates and shows a plot for suitable tasks/data.

    Returns:
        str: A summary of the results or an error message.
    """
    if model_params is None:
        model_params = {}

    # Validate task_name
    supported_tasks = ["classification", "regression", "clustering"]
    if task_name not in supported_tasks:
        return f"Unsupported task: '{task_name}'. Supported tasks are: {', '.join(supported_tasks)}."

    # Generate Data
    X, y = _generate_synthetic_data(task_name, data_params)
    if X is None:
        return "Error: Failed to generate data. Check data_params."

    # Train/Test Split (not for clustering)
    X_train, X_test, y_train, y_test = X, None, y, None
    X_plot, y_plot = X, y # Data to use for plotting
    plot_title_suffix = "(Full Dataset)"

    if task_name in ["classification", "regression"]:
        if y is None: # Should not happen if _generate_synthetic_data worked
             return f"Error: Target variable y is None for {task_name} after data generation."
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=data_params.get('random_state'))
        X_plot, y_plot = X_test, y_test # Prefer to plot test data results
        plot_title_suffix = "(Test Set)"


    # Adjust model_params for clustering if n_clusters not specified
    if task_name == "clustering" and 'n_clusters' not in model_params:
        model_params['n_clusters'] = data_params.get('centers', 3)
    if 'random_state' not in model_params and 'random_state' in data_params:
        model_params['random_state'] = data_params['random_state']


    # Train Model
    # For clustering, X_train is the full X dataset. y_train is None.
    model = _train_model(task_name, X_train, y_train, model_params)
    if model is None:
        return "Error: Failed to train model."

    # Evaluate Model
    # For clustering, evaluate on the full dataset (X_train which is X)
    eval_X = X_test if task_name in ["classification", "regression"] else X_train
    eval_y = y_test if task_name in ["classification", "regression"] else None # y_train is None for clustering

    metrics = _evaluate_model(task_name, model, eval_X, eval_y)
    if not metrics:
        return "Error: Failed to evaluate model."

    # Prepare Summary
    summary = f"--- {task_name.capitalize()} Task Summary ---\n"
    summary += f"Data Parameters: {data_params}\n"
    if model_params:
      summary += f"Model Parameters: {model_params}\n"

    if task_name == "classification":
        summary += f"Accuracy on Test Set: {metrics.get('accuracy', 'N/A'):.4f}\n"
    elif task_name == "regression":
        summary += f"R-squared on Test Set: {metrics.get('r2_score', 'N/A'):.4f}\n"
        summary += f"Coefficients: {metrics.get('coefficients', 'N/A')}\n"
    elif task_name == "clustering":
        summary += f"Inertia: {metrics.get('inertia', 'N/A'):.2f}\n"
        # summary += f"Cluster Centers:\n{metrics.get('cluster_centers', 'N/A')}\n" # Can be verbose

    # Plotting
    plot_generated_msg = ""
    if plot:
        if task_name == "clustering":
            _plot_results(task_name, model, X, model.labels_, title_suffix=plot_title_suffix) # Plot all data with labels
        elif task_name == "classification":
             # Plot test data with predicted labels for classification
            if X_test is not None and X_test.shape[1] == 2 :
                 _plot_results(task_name, model, X_test, model.predict(X_test), title_suffix=plot_title_suffix)
            elif X_train.shape[1] == 2 : # Fallback to train if test not suitable
                 _plot_results(task_name, model, X_train, model.predict(X_train), title_suffix="(Train Set Predictions)")
            else:
                 print("Plotting for classification skipped: Test data not 2D.")
        elif task_name == "regression":
            # Plot test data for regression
            if X_test is not None and X_test.shape[1] == 1:
                _plot_results(task_name, model, X_test, y_test, title_suffix=plot_title_suffix)
            elif X_train.shape[1] == 1: # Fallback to train if test not suitable
                _plot_results(task_name, model, X_train, y_train, title_suffix="(Train Set)")
            else:
                print("Plotting for regression skipped: Test data not 1D.")
        plot_generated_msg = "A plot has been generated (if data dimensions were suitable)."

    return summary + plot_generated_msg


# --- Example Usage ---
if __name__ == "__main__":
    print("Running Enhanced ML Task Examples...\n")

    # Common random state for examples
    rs = 42

    # --- Classification Example ---
    print("\n=== Classification Example (2 Features, 2 Classes) ===")
    class_data_params_2d = {'n_samples': 150, 'n_features': 2, 'n_classes': 2, 'random_state': rs}
    class_model_params = {'random_state': rs}
    class_summary_2d = run_ml_task("classification", class_data_params_2d, model_params=class_model_params, plot=True)
    print(class_summary_2d)

    print("\n=== Classification Example (4 Features, 3 Classes) ===")
    class_data_params_multi = {'n_samples': 200, 'n_features': 4, 'n_classes': 3, 'n_informative': 3, 'random_state': rs + 1}
    # Plotting will be skipped for >2 features by _plot_results
    class_summary_multi = run_ml_task("classification", class_data_params_multi, model_params={'random_state': rs + 1}, plot=True)
    print(class_summary_multi)

    # --- Regression Example ---
    print("\n=== Regression Example (1 Feature) ===")
    reg_data_params_1d = {'n_samples': 100, 'n_features': 1, 'noise': 20, 'random_state': rs + 2}
    reg_summary_1d = run_ml_task("regression", reg_data_params_1d, plot=True)
    print(reg_summary_1d)

    print("\n=== Regression Example (3 Features) ===")
    reg_data_params_multi = {'n_samples': 120, 'n_features': 3, 'noise': 15, 'random_state': rs + 3}
    # Plotting will be skipped for >1 feature by _plot_results
    reg_summary_multi = run_ml_task("regression", reg_data_params_multi, plot=True)
    print(reg_summary_multi)

    # --- Clustering Example ---
    print("\n=== Clustering Example (2 Features, 3 Centers) ===")
    cluster_data_params_2d = {'n_samples': 200, 'n_features': 2, 'centers': 3, 'cluster_std': 0.8, 'random_state': rs + 4}
    # KMeans n_clusters should match 'centers' in data_params
    cluster_model_params = {'n_clusters': 3, 'random_state': rs + 4}
    cluster_summary_2d = run_ml_task("clustering", cluster_data_params_2d, model_params=cluster_model_params, plot=True)
    print(cluster_summary_2d)

    print("\n=== Clustering Example (4 Features, 5 Centers) ===")
    cluster_data_params_multi = {'n_samples': 250, 'n_features': 4, 'centers': 5, 'cluster_std': 1.2, 'random_state': rs + 5}
    cluster_model_params_multi = {'n_clusters': 5, 'random_state': rs + 5}
    # Plotting will be skipped for >2 features by _plot_results
    cluster_summary_multi = run_ml_task("clustering", cluster_data_params_multi, model_params=cluster_model_params_multi, plot=True)
    print(cluster_summary_multi)

    # --- Unsupported Task Example ---
    print("\n=== Unsupported Task Example ===")
    unsupported_summary = run_ml_task("recommendation_system", {'n_samples': 50, 'random_state': rs})
    print(unsupported_summary)

    # --- Missing Params Example ---
    print("\n=== Missing Params Example ===")
    missing_params_summary = run_ml_task("classification", {'n_samples': 50}) # Missing 'random_state'
    print(missing_params_summary)

    print("\nNote: To use this module in Colab, first run in a cell:")
    print("# !pip install scikit-learn numpy pandas matplotlib")
    print("# Then, upload this .py file or copy its content into a cell, then import/use.")
    print("# Plots will be displayed inline in Colab notebooks.")
