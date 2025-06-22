# colab_ml_tools.py
# Enhanced module for common ML tasks in Colab.

import numpy as np
import pandas as pd # Not strictly used in current funcs, but good for future data handling
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs, make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, r2_score # Removed silhouette_score as it's not in direct requirements
from sklearn.preprocessing import StandardScaler

# Default parameters for data generation to ensure consistency and avoid magic numbers.
DEFAULT_DATA_PARAMS = {
    "classification": {'n_samples': 150, 'n_features': 2, 'n_classes': 2, 'n_informative': 2, 'random_state': 42},
    "regression": {'n_samples': 100, 'n_features': 1, 'n_informative': 1, 'noise': 10.0, 'random_state': 42},
    "clustering": {'n_samples': 200, 'n_features': 2, 'centers': 3, 'cluster_std': 1.0, 'random_state': 42},
}

DEFAULT_MODEL_PARAMS = {
    "classification": {'solver': 'liblinear', 'random_state': 42}, # liblinear is good for small datasets
    "regression": {}, # LinearRegression has fewer critical params for basic use
    "clustering": {'n_init': 'auto', 'random_state': 42}, # n_init='auto' to suppress warnings
}

# --- Helper Functions ---

def _generate_synthetic_data(task_name: str, data_params: dict):
    """
    Generates synthetic data for specified ML task using defaults if params not fully provided.
    Applies StandardScaler to the features.
    """
    # Merge provided data_params with defaults for the task
    # User-provided params will override defaults.
    current_data_params = {**DEFAULT_DATA_PARAMS.get(task_name, {}), **data_params}

    # Check for essential keys after merging, though defaults should cover them.
    required_keys = ['n_samples', 'random_state'] # n_features is also essential but handled by defaults
    if not all(key in current_data_params for key in required_keys):
        missing = [key for key in required_keys if key not in current_data_params]
        raise ValueError(f"Missing required data_params for {task_name}: {', '.join(missing)}. Current params: {current_data_params}")

    X, y_true = None, None # y_true is for true labels/values, y for clustering is different

    if task_name == "classification":
        # Ensure n_informative is not greater than n_features
        if current_data_params.get('n_informative', 2) > current_data_params.get('n_features', 2):
            current_data_params['n_informative'] = current_data_params['n_features']
        # Ensure n_redundant + n_informative <= n_features
        n_red = current_data_params.get('n_features', 2) - current_data_params.get('n_informative', 2)

        X, y_true = make_classification(
            n_samples=current_data_params['n_samples'],
            n_features=current_data_params.get('n_features', 2),
            n_informative=current_data_params.get('n_informative', 2),
            n_redundant=max(0, n_red), # Ensure non-negative
            n_classes=current_data_params.get('n_classes', 2),
            random_state=current_data_params['random_state']
        )
    elif task_name == "regression":
        if current_data_params.get('n_informative', 1) > current_data_params.get('n_features', 1):
            current_data_params['n_informative'] = current_data_params['n_features']
        X, y_true = make_regression(
            n_samples=current_data_params['n_samples'],
            n_features=current_data_params.get('n_features', 1),
            n_informative=current_data_params.get('n_informative', 1),
            noise=current_data_params.get('noise', 10.0),
            random_state=current_data_params['random_state']
        )
    elif task_name == "clustering":
        # For make_blobs, y_true will contain the true cluster assignments
        X, y_true = make_blobs(
            n_samples=current_data_params['n_samples'],
            n_features=current_data_params.get('n_features', 2),
            centers=current_data_params.get('centers', 3),
            cluster_std=current_data_params.get('cluster_std', 1.0),
            random_state=current_data_params['random_state']
        )
        # For clustering, the run_ml_task expects (X, None) from this helper,
        # as y is not used for training KMeans directly (it's unsupervised).
        # The y_true from make_blobs can be used for external evaluation if needed, but not by _train_model.
        # We will return X and the y_true (cluster assignments) for potential use in plotting true clusters.
        # The main run_ml_task will handle passing None to _train_model for y_train in clustering.
    else:
        # This case should ideally be caught by the main function's task validation
        raise ValueError(f"Unknown task_name '{task_name}' in _generate_synthetic_data.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y_true


def _train_model(task_name: str, X_train: np.ndarray, y_train: np.ndarray = None, model_params: dict = None):
    """Initializes and trains a model, using defaults if params not fully provided."""
    current_model_params = {**DEFAULT_MODEL_PARAMS.get(task_name, {}), **(model_params if model_params else {})}

    model = None
    if task_name == "classification":
        model = LogisticRegression(**current_model_params)
        if y_train is None: raise ValueError("y_train cannot be None for classification.")
        model.fit(X_train, y_train)
    elif task_name == "regression":
        model = LinearRegression(**current_model_params)
        if y_train is None: raise ValueError("y_train cannot be None for regression.")
        model.fit(X_train, y_train)
    elif task_name == "clustering":
        # Ensure n_clusters is set, defaulting from data_params if not in model_params
        if 'n_clusters' not in current_model_params:
             # This assumes data_params passed to run_ml_task had 'centers'
             # or DEFAULT_DATA_PARAMS for clustering was used.
             current_model_params['n_clusters'] = DEFAULT_DATA_PARAMS['clustering'].get('centers',3)

        model = KMeans(**current_model_params)
        model.fit(X_train) # y_train is not used for KMeans
    else:
        raise ValueError(f"Unknown task_name '{task_name}' in _train_model.")
    return model


def _evaluate_model(task_name: str, model, X_eval: np.ndarray, y_eval: np.ndarray = None):
    """Evaluates the trained model."""
    metrics = {}
    if task_name == "classification":
        if y_eval is None: raise ValueError("y_eval cannot be None for classification evaluation.")
        predictions = model.predict(X_eval)
        metrics["accuracy"] = accuracy_score(y_eval, predictions)
    elif task_name == "regression":
        if y_eval is None: raise ValueError("y_eval cannot be None for regression evaluation.")
        metrics["r2_score"] = model.score(X_eval, y_eval)
        metrics["coefficients"] = model.coef_
        if hasattr(model, 'intercept_'): metrics["intercept"] = model.intercept_
    elif task_name == "clustering":
        metrics["inertia"] = model.inertia_
        if hasattr(model, 'cluster_centers_'): metrics["cluster_centers_shape"] = model.cluster_centers_.shape
    else:
        raise ValueError(f"Unknown task_name '{task_name}' in _evaluate_model.")
    return metrics


def _plot_results(task_name: str, model, X: np.ndarray, y_true_or_predicted_labels: np.ndarray = None, X_test: np.ndarray = None, y_test_true_labels: np.ndarray = None, title_suffix: str = ""):
    """Generates and displays a plot for the task results. Handles 1D/2D features."""
    if not ( (X.shape[1] == 1 and task_name == "regression") or \
             (X.shape[1] == 2 and task_name in ["classification", "clustering"]) ):
        print(f"Plotting for {task_name} is only supported for 1D features (regression) or 2D features (classification/clustering). Data has {X.shape[1]} features.")
        plt.close('all') # Close any existing figures to prevent display of empty plot
        return False # Indicate plotting was not performed

    plt.figure(figsize=(8, 6))
    title = f"{task_name.capitalize()} Results {title_suffix}"
    plot_made = False

    if task_name == "classification":
        if X.shape[1] == 2 and y_true_or_predicted_labels is not None:
            plt.scatter(X[:, 0], X[:, 1], c=y_true_or_predicted_labels, cmap=plt.cm.Paired, edgecolors='k', alpha=0.7, label="Data Points (colored by label/prediction)")
            plot_made = True
        # Could add decision boundary plotting here for specific models like LogisticRegression if time permits.

    elif task_name == "regression":
        if X.shape[1] == 1 and y_true_or_predicted_labels is not None:
            plt.scatter(X, y_true_or_predicted_labels, color='blue', label='Actual data', edgecolors='k', alpha=0.7)
            # Sort X for smooth line plotting if it's not already sorted
            sort_indices = X[:,0].argsort()
            X_sorted = X[sort_indices]
            plt.plot(X_sorted, model.predict(X_sorted), color='red', linewidth=2, label='Regression line')
            plt.legend()
            plot_made = True

    elif task_name == "clustering":
        if X.shape[1] == 2:
            # y_true_or_predicted_labels here are the predicted cluster labels from model.labels_
            predicted_labels = model.labels_
            centers = model.cluster_centers_
            plt.scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='viridis', edgecolors='k', s=50, alpha=0.7, label="Data Points (colored by predicted cluster)")
            if centers is not None and centers.shape[1] == 2 :
                 plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.9, marker='X', label='Centroids')
            plt.legend()
            plot_made = True

    if plot_made:
        plt.title(title)
        plt.xlabel("Feature 1" if X.shape[1] > 0 else "Feature")
        if X.shape[1] > 1: plt.ylabel("Feature 2")
        plt.grid(True)
        plt.show()
        return True
    else:
        plt.close('all') # Ensure no empty plot is shown if conditions not met
        return False

# --- Core Function ---

def run_ml_task(task_name: str, data_params: dict, model_params: dict = None, plot: bool = False) -> str:
    """
    Runs a specified ML task: data generation, scaling, model training, evaluation, and optional plotting.

    Args:
        task_name (str): "classification", "regression", or "clustering".
        data_params (dict): Parameters for synthetic data generation. Overrides defaults.
        model_params (dict, optional): Parameters for model initialization. Overrides defaults.
        plot (bool, optional): If True, generates and shows a plot for suitable tasks/data.

    Returns:
        str: A summary of the results or an error message.
    """
    supported_tasks = ["classification", "regression", "clustering"]
    if task_name not in supported_tasks:
        return f"Error: Unsupported task '{task_name}'. Supported tasks are: {', '.join(supported_tasks)}."

    if model_params is None: model_params = {}

    final_data_params = {**DEFAULT_DATA_PARAMS.get(task_name, {}), **data_params}
    final_model_params = {**DEFAULT_MODEL_PARAMS.get(task_name, {}), **model_params}


    try:
        # Generate Data (X is scaled, y_data is true labels/values or true cluster IDs for clustering)
        X_scaled, y_data = _generate_synthetic_data(task_name, final_data_params)

        X_train, X_test, y_train, y_test = X_scaled, None, y_data, None

        if task_name in ["classification", "regression"]:
            if y_data is None: raise ValueError(f"Target variable y_data is None for {task_name}.")
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_data, test_size=0.3, random_state=final_data_params.get('random_state')
            )
        elif task_name == "clustering":
            # For clustering, train on full dataset. y_train for _train_model should be None.
            X_train, y_train = X_scaled, None
            # X_test, y_test remain None, evaluation is on the full X_scaled via inertia.
            # y_data from _generate_synthetic_data (true cluster IDs) can be used for plotting if desired,
            # but is not part of the standard KMeans training/evaluation flow here.

        # Adjust model_params for clustering n_clusters if not set by user, based on data_params's 'centers'
        if task_name == "clustering" and 'n_clusters' not in final_model_params:
            final_model_params['n_clusters'] = final_data_params.get('centers', DEFAULT_DATA_PARAMS['clustering']['centers'])

        # Ensure random_state for model is consistent if not explicitly set in model_params
        if 'random_state' not in final_model_params and 'random_state' in final_data_params:
             final_model_params['random_state'] = final_data_params['random_state']


        model = _train_model(task_name, X_train, y_train, final_model_params)

        # Evaluation
        eval_X = X_test if task_name in ["classification", "regression"] else X_train # Use X_train (full data) for clustering inertia
        eval_y = y_test if task_name in ["classification", "regression"] else None # No y_true for clustering internal eval
        metrics = _evaluate_model(task_name, model, eval_X, eval_y)

        # Prepare Summary
        summary = f"--- {task_name.capitalize()} Task Summary ---\n"
        summary += f"Data Generation Parameters (merged with defaults): {final_data_params}\n"
        summary += f"Model Parameters (merged with defaults): {final_model_params}\n"

        if task_name == "classification":
            summary += f"Accuracy on Test Set: {metrics.get('accuracy', 'N/A'):.4f}\n"
        elif task_name == "regression":
            summary += f"R-squared on Test Set: {metrics.get('r2_score', 'N/A'):.4f}\n"
            summary += f"Coefficients: {metrics.get('coefficients', 'N/A')}\n"
            if "intercept" in metrics: summary += f"Intercept: {metrics.get('intercept', 'N/A')}\n"
        elif task_name == "clustering":
            summary += f"Inertia: {metrics.get('inertia', 'N/A'):.2f}\n"
            summary += f"Cluster Centers Shape: {metrics.get('cluster_centers_shape', 'N/A')}\n"

        plot_generated_msg = ""
        if plot:
            plot_title_suffix = ""
            data_for_plot_X = X_scaled # Default to full dataset for plotting context
            data_for_plot_y = y_data   # True labels/values from generation, or predicted for clustering

            if task_name == "classification":
                plot_title_suffix = "(Test Set Predictions)"
                data_for_plot_X = X_test if X_test is not None else X_train
                data_for_plot_y = model.predict(data_for_plot_X) if data_for_plot_X is not None else None
            elif task_name == "regression":
                plot_title_suffix = "(Test Set)"
                data_for_plot_X = X_test if X_test is not None else X_train
                data_for_plot_y = y_test if y_test is not None else y_train # True y values for scatter
            elif task_name == "clustering":
                plot_title_suffix = "(Predicted Clusters on Full Dataset)"
                # data_for_plot_X is already X_scaled (full dataset)
                # data_for_plot_y will be model.labels_ inside _plot_results

            plot_successful = _plot_results(task_name, model, data_for_plot_X, data_for_plot_y, title_suffix=plot_title_suffix)
            if plot_successful:
                plot_generated_msg = "Plot generated successfully."
            else:
                plot_generated_msg = "Plotting skipped (e.g., data not 1D/2D or other issue)."

        return summary + plot_generated_msg

    except ValueError as ve:
        return f"Error in ML task '{task_name}': {ve}"
    except Exception as e:
        return f"An unexpected error occurred in ML task '{task_name}': {type(e).__name__} - {e}"


# --- Example Usage ---
if __name__ == "__main__":
    print("Running Enhanced ML Task Examples (colab_ml_tools.py)...\n")

    rs = 42 # Common random state for examples

    # --- Classification Example ---
    print("\n=== Classification Example (Defaults, Plot) ===")
    # Uses mostly defaults from DEFAULT_DATA_PARAMS and DEFAULT_MODEL_PARAMS
    class_summary_def = run_ml_task("classification", {'random_state': rs}, plot=True)
    print(class_summary_def)

    print("\n=== Classification Example (Custom Params, 4 Features, 3 Classes, No Plot) ===")
    class_data_custom = {'n_samples': 200, 'n_features': 4, 'n_classes': 3, 'n_informative': 3, 'random_state': rs + 1}
    class_model_custom = {'solver': 'saga', 'C': 0.5, 'random_state': rs + 1} # Example of different solver
    class_summary_custom = run_ml_task("classification", class_data_custom, model_params=class_model_custom, plot=False)
    print(class_summary_custom)


    # --- Regression Example ---
    print("\n=== Regression Example (Defaults, Plot) ===")
    reg_summary_def = run_ml_task("regression", {'random_state': rs + 2, 'noise': 25}, plot=True) # Override default noise
    print(reg_summary_def)

    print("\n=== Regression Example (3 Features, No Plot) ===")
    reg_data_3feat = {'n_samples': 120, 'n_features': 3, 'n_informative': 2, 'noise': 15, 'random_state': rs + 3}
    reg_summary_3feat = run_ml_task("regression", reg_data_3feat, plot=False)
    print(reg_summary_3feat)


    # --- Clustering Example ---
    print("\n=== Clustering Example (Defaults, Plot) ===")
    cluster_summary_def = run_ml_task("clustering", {'random_state': rs + 4, 'centers': 4}, plot=True) # Override default centers
    print(cluster_summary_def)

    print("\n=== Clustering Example (Custom Params, 5 Centers, No Plot) ===")
    cluster_data_custom = {'n_samples': 250, 'n_features': 2, 'centers': 5, 'cluster_std': 1.5, 'random_state': rs + 5}
    # KMeans n_clusters will be inferred from data_params' 'centers' if not in model_params
    cluster_model_custom = {'n_init': 10, 'random_state': rs + 5} # explicit n_init
    cluster_summary_custom = run_ml_task("clustering", cluster_data_custom, model_params=cluster_model_custom, plot=False)
    print(cluster_summary_custom)


    # --- Error Handling Examples ---
    print("\n=== Unsupported Task Example ===")
    unsupported_summary = run_ml_task("recommendation", {'n_samples': 50, 'random_state': rs})
    print(unsupported_summary)

    print("\n=== Missing Essential Params Example (if defaults didn't cover) ===")
    # Modify DEFAULT_DATA_PARAMS temporarily to simulate missing 'random_state' for testing error
    # This is tricky because DEFAULT_DATA_PARAMS is global. For a real test, might need more setup.
    # For now, assume _generate_synthetic_data's checks would catch it if defaults were incomplete.
    # The current implementation relies on defaults to provide essential params if user doesn't.
    # Let's test by providing an empty data_params dict for a task, relying on defaults.
    print("Testing with empty data_params (should use defaults):")
    empty_params_summary = run_ml_task("classification", {}, plot=False)
    print(empty_params_summary)

    print("\n=== Plotting with Unsuitable Dimensions (e.g., 3D classification) ===")
    class_3d_plot = run_ml_task("classification", {'n_features': 3, 'random_state': rs}, plot=True)
    print(class_3d_plot)


    print("\nNote: To use this module in Colab, ensure matplotlib is available.")
    print("# Plots will be displayed inline in Colab notebooks if `%matplotlib inline` is set.")
