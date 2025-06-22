# aethermind_main.py
# Central controller and task router for AetherMindAI

import sys

# Attempt to import necessary modules and set availability flags
phi2_available = False
math_solver_available = False
ml_tools_available = False

try:
    from phi2_colab_runner import generate_response, model, tokenizer
    phi2_available = True
    print("Successfully imported phi2_colab_runner.")
except ImportError:
    print("Error: Failed to import 'phi2_colab_runner'. Chat functionality will be unavailable.")
    # To allow the script to run for other tests, we might not exit immediately
    # but phi2_available will remain False.
    # For the main loop to function, phi2 is essential.
    model, tokenizer = None, None # Ensure these are defined even if import fails

try:
    from colab_math_solver import solve_math_query
    math_solver_available = True
    print("Successfully imported colab_math_solver.")
except ImportError:
    print("Error: Failed to import 'colab_math_solver'. Math solving functionality will be unavailable.")

try:
    from colab_ml_tools import run_ml_task
    ml_tools_available = True
    print("Successfully imported colab_ml_tools.")
except ImportError:
    print("Error: Failed to import 'colab_ml_tools'. ML task functionality will be unavailable.")

# Placeholder for further code
if __name__ == "__main__":
    if not phi2_available:
        print("Core chat module (phi2_colab_runner) is not available. AetherMind cannot start.")
        sys.exit(1)

    print("\nAetherMindAI Controller Initialized.")
    print(f"Phi2 Chat: {'Available' if phi2_available else 'Unavailable'}")
    print(f"Math Solver: {'Available' if math_solver_available else 'Unavailable'}")
    print(f"ML Tools: {'Available' if ml_tools_available else 'Unavailable'}")
    # print("\nFurther implementation will go here (task detection, chat loop, etc.)") # Placeholder removed

# --- Task Detection Logic ---
MATH_KEYWORDS = [
    "solve", "calculate", "computation", "equation", "derivative", "integrate", "integral",
    "limit", "matrix", "algebra", "factor", "expand", "polynomial", "arithmetic",
    "geometry problem", "math problem", "differentiate"
]
ML_KEYWORDS = [
    "classify", "classification", "categorize", "predict category",
    "regression", "predict value", "forecast",
    "cluster", "clustering", "group data", "segmentation", "k-means", "kmeans",
    "train model", "machine learning", "ml model", "ai model", "neural network",
    "decision tree", "logistic regression", "linear regression", "svm",
    "dataset", "feature", "label", "accuracy", "precision", "recall", "f1-score",
    "run ml", "perform analysis"
]

def detect_task_type(prompt: str) -> str:
    """
    Classifies the user's prompt into 'math', 'ml', or 'chat'.

    Args:
        prompt (str): The user input string.

    Returns:
        str: The detected task type.
    """
    prompt_lower = prompt.lower()

    if any(keyword in prompt_lower for keyword in MATH_KEYWORDS):
        return "math"

    if any(keyword in prompt_lower for keyword in ML_KEYWORDS):
        return "ml"

    return "chat"


if __name__ == "__main__":
    if not phi2_available:
        print("Core chat module (phi2_colab_runner) is not available. AetherMind cannot start.")
        sys.exit(1)

    print("\nAetherMindAI Controller Initialized.")
    print(f"Phi2 Chat: {'Available' if phi2_available else 'Unavailable'}")
    print(f"Math Solver: {'Available' if math_solver_available else 'Unavailable'}")
    print(f"ML Tools: {'Available' if ml_tools_available else 'Unavailable'}")

    # --- Test detect_task_type ---
    print("\n--- Testing Task Detection ---")
    test_prompts_detection = {
        "Solve x^2 - 4 = 0": "math",
        "Can you run a classification task for me?": "ml",
        "Tell me a joke.": "chat",
        "What is the integral of 2x?": "math",
        "I need to perform clustering on a dataset.": "ml",
        "What's the weather like?": "chat",
        "Calculate the limit of (sin x)/x as x approaches 0": "math",
        "Train a regression model to predict house prices": "ml"
    }
    for p, expected in test_prompts_detection.items():
        detected = detect_task_type(p)
        print(f"Prompt: \"{p}\" | Expected: {expected} | Detected: {detected} | Correct: {detected == expected}")

    # print("\nFurther implementation for ML subtask detection, chat loop, etc., will follow.") # Placeholder removed

# --- ML Sub-task Detection and Parameter Generation ---
ML_CLASSIFICATION_KEYWORDS = ["classification", "classify", "categorize", "logistic regression", "decision tree"]
ML_REGRESSION_KEYWORDS = ["regression", "predict value", "forecast", "linear regression"]
ML_CLUSTERING_KEYWORDS = ["clustering", "cluster", "group data", "segmentation", "k-means", "kmeans"]

def get_ml_subtask_and_params(prompt: str) -> tuple[str | None, dict | None]:
    """
    Infers ML sub-task from prompt and returns task name and default data parameters.

    Args:
        prompt (str): The user input string (expected to be pre-identified as an "ml" task).

    Returns:
        tuple[str | None, dict | None]:
            - ML task name string ("classification", "regression", "clustering") or None if not specific.
            - Dictionary of default data parameters, or None if task_name is None.
    """
    prompt_lower = prompt.lower()

    # Default data parameters, can be customized per sub-task if needed
    default_data_params = {
        'n_samples': 150,  # Increased samples slightly
        'random_state': 42
    }

    task_name = None

    if any(keyword in prompt_lower for keyword in ML_CLASSIFICATION_KEYWORDS):
        task_name = "classification"
        default_data_params['n_features'] = data_params_ml.get('n_features', 4) # Default for classification
        default_data_params['n_classes'] = data_params_ml.get('n_classes', 2)
        default_data_params['n_informative'] = data_params_ml.get('n_informative', 2)

    elif any(keyword in prompt_lower for keyword in ML_REGRESSION_KEYWORDS):
        task_name = "regression"
        default_data_params['n_features'] = data_params_ml.get('n_features', 1) # Default for regression (plotable)
        default_data_params['n_informative'] = data_params_ml.get('n_informative', 1)
        default_data_params['noise'] = data_params_ml.get('noise', 10.0)

    elif any(keyword in prompt_lower for keyword in ML_CLUSTERING_KEYWORDS):
        task_name = "clustering"
        default_data_params['n_features'] = data_params_ml.get('n_features', 2) # Default for clustering (plotable)
        default_data_params['centers'] = data_params_ml.get('centers', 3)
        default_data_params['cluster_std'] = data_params_ml.get('cluster_std', 1.0)

    if task_name:
        return task_name, default_data_params
    else:
        # If no specific sub-task keywords, but was detected as "ml" generally
        # We can default to one, or return None to let the caller decide/fallback
        # Defaulting to "classification" for now if it's an ML prompt but no sub-task found
        # print("No specific ML sub-task identified, defaulting to classification.")
        # default_data_params['n_features'] = 2
        # default_data_params['n_classes'] = 2
        # default_data_params['n_informative'] = 2
        # return "classification", default_data_params
        return None, None # Let caller handle ambiguity

# Dummy data_params_ml for testing get_ml_subtask_and_params structure
# In real use, these might come from a config or more advanced parsing
data_params_ml = {
    'n_samples': 100,
    'n_features': 2,
    'random_state': 42,
    'n_classes': 3,
    'n_informative': 2,
    'noise': 0.1,
    'centers': 3,
    'cluster_std': 1.0
}

if __name__ == "__main__":
    if not phi2_available:
        print("Core chat module (phi2_colab_runner) is not available. AetherMind cannot start.")
        sys.exit(1)

    print("\nAetherMindAI Controller Initialized.")
    print(f"Phi2 Chat: {'Available' if phi2_available else 'Unavailable'}")
    print(f"Math Solver: {'Available' if math_solver_available else 'Unavailable'}")
    print(f"ML Tools: {'Available' if ml_tools_available else 'Unavailable'}")

    # --- Test detect_task_type ---
    print("\n--- Testing Task Detection ---")
    test_prompts_detection = {
        "Solve x^2 - 4 = 0": "math",
        "Can you run a classification task for me?": "ml",
        "Tell me a joke.": "chat",
        "What is the integral of 2x?": "math",
        "I need to perform clustering on a dataset.": "ml",
        "What's the weather like?": "chat",
        "Calculate the limit of (sin x)/x as x approaches 0": "math",
        "Train a regression model to predict house prices": "ml"
    }
    for p, expected in test_prompts_detection.items():
        detected = detect_task_type(p)
        print(f"Prompt: \"{p}\" | Expected: {expected} | Detected: {detected} | Correct: {detected == expected}")

    # --- Test get_ml_subtask_and_params ---
    print("\n--- Testing ML Sub-task Detection ---")
    test_ml_prompts = {
        "Run a classification model.": ("classification", True),
        "I want to do some regression analysis.": ("regression", True),
        "Perform k-means clustering.": ("clustering", True),
        "Train an ML model.": (None, False), # Ambiguous
        "Is logistic regression good for this?": ("classification", True)
    }
    for p_ml, (expected_task, params_expected) in test_ml_prompts.items():
        task, params = get_ml_subtask_and_params(p_ml)
        print(f"Prompt: \"{p_ml}\" | Expected Task: {expected_task} | Detected Task: {task} | Params Expected: {params_expected} | Params Returned: {params is not None}")
        if params:
            print(f"      Params: {params}")

    # print("\nFurther implementation for chat loop, etc., will follow.") # Placeholder removed


# --- Main Chat Loop ---
def aethermind_chat_loop():
    """
    Main interactive chat loop for AetherMindAI.
    Routes prompts to appropriate handlers based on detected task type.
    """
    if not phi2_available:
        print("Critical Error: Phi-2 chat model (phi2_colab_runner) is not available. Cannot start chat loop.")
        return

    print("\nWelcome to AetherMindAI!")
    print("Type 'exit' or 'quit' to end the session.")
    print("Modules loaded:")
    print(f"  - Phi2 Chat: {'Available' if phi2_available else 'Unavailable (Critical!)'}")
    print(f"  - Math Solver: {'Available' if math_solver_available else 'Unavailable'}")
    print(f"  - ML Tools: {'Available' if ml_tools_available else 'Unavailable'}")
    print("-" * 30)

    history = []

    while True:
        try:
            prompt = input("You: ")
            if prompt.lower() in ["exit", "quit"]:
                print("AetherMindAI: Goodbye!")
                break

            task_type = detect_task_type(prompt)
            response = ""
            response_label = "[AetherMind Chat]" # Default label

            if task_type == "math":
                if math_solver_available:
                    try:
                        math_result = solve_math_query(prompt)
                        response = math_result
                        response_label = "[Math Solver]"
                    except Exception as e:
                        response = f"Math solver encountered an error: {e}. Falling back to chat."
                        # Fallback to chat if math solver fails
                        if phi2_available and model and tokenizer:
                             response += "\n[AetherMind Chat]: " + generate_response(prompt, history, model, tokenizer)
                        else:
                             response += "\n[AetherMind Chat]: Chat model unavailable for fallback."
                        response_label = "[Error/Chat Fallback]"
                else:
                    response = "Math solver module is not available. I'll try to answer as a chatbot."
                    if phi2_available and model and tokenizer:
                        response += "\n[AetherMind Chat]: " + generate_response(prompt, history, model, tokenizer)
                    else:
                        response += "\n[AetherMind Chat]: Chat model unavailable."
                    response_label = "[Chat Fallback]"

            elif task_type == "ml":
                if ml_tools_available:
                    ml_task_name, ml_data_params = get_ml_subtask_and_params(prompt)
                    if ml_task_name and ml_data_params:
                        try:
                            # For now, plot=False in the loop for console simplicity
                            ml_result = run_ml_task(task_name=ml_task_name, data_params=ml_data_params, plot=False)
                            response = ml_result
                            response_label = f"[ML Tool: {ml_task_name.capitalize()}]"
                        except Exception as e:
                            response = f"ML tool encountered an error: {e}. Falling back to chat."
                            if phi2_available and model and tokenizer:
                                response += "\n[AetherMind Chat]: " + generate_response(prompt, history, model, tokenizer)
                            else:
                                response += "\n[AetherMind Chat]: Chat model unavailable for fallback."
                            response_label = "[Error/Chat Fallback]"
                    else:
                        response = "Could not determine a specific ML task from your prompt. Trying general chat."
                        if phi2_available and model and tokenizer:
                             response += "\n[AetherMind Chat]: " + generate_response(prompt, history, model, tokenizer)
                        else:
                             response += "\n[AetherMind Chat]: Chat model unavailable."
                        response_label = "[Chat Fallback]"
                else:
                    response = "ML tools module is not available. I'll try to answer as a chatbot."
                    if phi2_available and model and tokenizer:
                        response += "\n[AetherMind Chat]: " + generate_response(prompt, history, model, tokenizer)
                    else:
                        response += "\n[AetherMind Chat]: Chat model unavailable."
                    response_label = "[Chat Fallback]"

            else: # Default to "chat"
                if phi2_available and model and tokenizer:
                    try:
                        chat_response = generate_response(prompt, history, model, tokenizer)
                        response = chat_response
                        # response_label is already "[AetherMind Chat]"
                    except Exception as e:
                        response = f"Chat model encountered an error: {e}."
                        response_label = "[Error]"
                else:
                    response = "Chat model is not available. Cannot process this request."
                    response_label = "[Error]"

            print(f"{response_label}: {response}")
            history.append({'role': 'user', 'content': prompt})
            history.append({'role': 'assistant', 'content': f"{response_label}: {response}"}) # Store full response with label

            # Keep history to a manageable size (e.g., last 5 interactions = 10 entries)
            if len(history) > 10:
                history = history[-10:]

        except KeyboardInterrupt:
            print("\nAetherMindAI: Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"An unexpected error occurred in the chat loop: {e}")
            # Optionally, decide if to break or continue
            # break


if __name__ == "__main__":
    if not phi2_available:
        print("Core chat module (phi2_colab_runner) is not available. AetherMind cannot start.")
        sys.exit(1)

    print("\nAetherMindAI Controller Initialized.")
    print(f"Phi2 Chat: {'Available' if phi2_available else 'Unavailable'}")
    print(f"Math Solver: {'Available' if math_solver_available else 'Unavailable'}")
    print(f"ML Tools: {'Available' if ml_tools_available else 'Unavailable'}")

    # --- Test detect_task_type ---
    print("\n--- Testing Task Detection ---")
    test_prompts_detection = {
        "Solve x^2 - 4 = 0": "math",
        "Can you run a classification task for me?": "ml",
        "Tell me a joke.": "chat",
        "What is the integral of 2x?": "math",
        "I need to perform clustering on a dataset.": "ml",
        "What's the weather like?": "chat",
        "Calculate the limit of (sin x)/x as x approaches 0": "math",
        "Train a regression model to predict house prices": "ml"
    }
    for p, expected in test_prompts_detection.items():
        detected = detect_task_type(p)
        print(f"Prompt: \"{p}\" | Expected: {expected} | Detected: {detected} | Correct: {detected == expected}")

    # --- Test get_ml_subtask_and_params ---
    print("\n--- Testing ML Sub-task Detection ---")
    test_ml_prompts = {
        "Run a classification model.": ("classification", True),
        "I want to do some regression analysis.": ("regression", True),
        "Perform k-means clustering.": ("clustering", True),
        "Train an ML model.": (None, False), # Ambiguous
        "Is logistic regression good for this?": ("classification", True)
    }
    for p_ml, (expected_task, params_expected) in test_ml_prompts.items():
        task, params = get_ml_subtask_and_params(p_ml)
        print(f"Prompt: \"{p_ml}\" | Expected Task: {expected_task} | Detected Task: {task} | Params Expected: {params_expected} | Params Returned: {params is not None}")
        if params:
            print(f"      Params: {params}")

    # --- Ask to run interactive loop ---
    print("\n--- Interactive Mode ---")
    run_loop = input("Do you want to start the interactive AetherMindAI chat loop? (yes/no): ")
    if run_loop.lower() == 'yes':
        aethermind_chat_loop()
    else:
        print("Skipping interactive chat loop. End of script.")
