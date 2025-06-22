# aethermind_main.py
# Central controller and task router for AetherMindAI

import sys

# --- Module Availability Flags & Initializations ---
phi2_model = None
phi2_tokenizer = None
phi2_available = False
math_solver_available = False
ml_tools_available = False

# Attempt to import LLM related modules first
try:
    from phi2_colab_runner import load_model_and_tokenizer, generate_response
    # Load model and tokenizer at startup
    print("Initializing Phi-2 model and tokenizer...")
    phi2_model, phi2_tokenizer = load_model_and_tokenizer()
    if phi2_model and phi2_tokenizer:
        phi2_available = True
        print("Phi-2 model and tokenizer loaded successfully.")
    else:
        print("Error: Failed to load Phi-2 model or tokenizer. Chat functionality will be impaired.")
except ImportError:
    print("Error: Failed to import 'phi2_colab_runner'. Chat functionality will be unavailable.")
except Exception as e:
    print(f"An unexpected error occurred during Phi-2 initialization: {e}")
    print("Chat functionality will be impaired.")


# Attempt to import other modules
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


# --- Task Detection Logic ---
MATH_KEYWORDS = [
    "solve", "calculate", "computation", "equation", "derivative", "integrate", "integral",
    "limit", "matrix", "algebra", "factor", "expand", "polynomial", "arithmetic",
    "geometry problem", "math problem", "differentiate", "series"
]
ML_KEYWORDS = [
    "classify", "classification", "categorize", "predict category",
    "regression", "predict value", "forecast",
    "cluster", "clustering", "group data", "segmentation", "k-means", "kmeans",
    "train model", "machine learning", "ml model", "ai model", "neural network",
    "decision tree", "logistic regression", "linear regression", "svm",
    "dataset", "feature", "label", "accuracy", "precision", "recall", "f1-score",
    "run ml", "perform analysis", "ml task"
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
        # Add more specific checks if needed to differentiate from general chat
        # For example, presence of numbers, operators, or specific math function names
        # Basic check: if it contains "solve for", "derivative of", "integral of" etc.
        if re.search(r"(solve|diff|integrate|limit|series|factor|expand)\s*\(", prompt_lower) or \
           re.search(r"solve\s+.*for\s+\w+", prompt_lower) or \
           re.search(r"derivative\s+of", prompt_lower) or \
           re.search(r"integral\s+of", prompt_lower):
            return "math"
        # If a general math keyword is present but not a clear command, could be ambiguous.
        # For now, any math keyword triggers "math".
        return "math"


    if any(keyword in prompt_lower for keyword in ML_KEYWORDS):
        return "ml"

    return "chat" # Default

import re # Needed for the refined math keyword check

# --- ML Sub-task Detection and Parameter Generation ---
ML_CLASSIFICATION_KEYWORDS = ["classification", "classify", "categorize", "logistic regression", "decision tree"]
ML_REGRESSION_KEYWORDS = ["regression", "predict value", "forecast", "linear regression"]
ML_CLUSTERING_KEYWORDS = ["clustering", "cluster", "group data", "segmentation", "k-means", "kmeans"]

# Dummy data_params_ml for defining structure expected by get_ml_subtask_and_params
# In real use, these might come from a config or more advanced parsing for default values
# This is just to prevent NameError if data_params_ml was accessed before assignment
# (though the current get_ml_subtask_and_params doesn't use it that way anymore)
data_params_ml_defaults = {
    'n_features_classification': 4, 'n_classes_classification': 2, 'n_informative_classification': 2,
    'n_features_regression': 1, 'n_informative_regression': 1, 'noise_regression': 10.0,
    'n_features_clustering': 2, 'centers_clustering': 3, 'cluster_std_clustering': 1.0
}


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

    default_data_params = {
        'n_samples': 150,
        'random_state': 42
    }
    task_name = None

    if any(keyword in prompt_lower for keyword in ML_CLASSIFICATION_KEYWORDS):
        task_name = "classification"
        default_data_params['n_features'] = data_params_ml_defaults.get('n_features_classification', 4)
        default_data_params['n_classes'] = data_params_ml_defaults.get('n_classes_classification', 2)
        default_data_params['n_informative'] = data_params_ml_defaults.get('n_informative_classification', 2)
    elif any(keyword in prompt_lower for keyword in ML_REGRESSION_KEYWORDS):
        task_name = "regression"
        default_data_params['n_features'] = data_params_ml_defaults.get('n_features_regression', 1)
        default_data_params['n_informative'] = data_params_ml_defaults.get('n_informative_regression', 1)
        default_data_params['noise'] = data_params_ml_defaults.get('noise_regression', 10.0)
    elif any(keyword in prompt_lower for keyword in ML_CLUSTERING_KEYWORDS):
        task_name = "clustering"
        default_data_params['n_features'] = data_params_ml_defaults.get('n_features_clustering', 2)
        default_data_params['centers'] = data_params_ml_defaults.get('centers_clustering', 3)
        default_data_params['cluster_std'] = data_params_ml_defaults.get('cluster_std_clustering', 1.0)

    if task_name:
        return task_name, default_data_params
    else:
        # If "ml task" was in prompt but no specific sub-task, could default or ask user.
        # For now, returning None, None to indicate ambiguity to the caller.
        if "ml task" in prompt_lower or "perform analysis" in prompt_lower: # General ML request
             print("Ambiguous ML task. Please specify: classification, regression, or clustering.")
        return None, None


# --- Main Chat Loop ---
def aethermind_chat_loop():
    """
    Main interactive chat loop for AetherMindAI.
    Routes prompts to appropriate handlers based on detected task type.
    Uses the globally loaded phi2_model and phi2_tokenizer.
    """
    if not phi2_available or not phi2_model or not phi2_tokenizer:
        print("Critical Error: Phi-2 model/tokenizer is not available. Cannot start chat loop.")
        return

    print("\nWelcome to AetherMindAI!")
    print("Type 'exit' or 'quit' to end the session.")
    print("Modules loaded:")
    print(f"  - Phi2 Chat: {'Available' if phi2_available else 'Unavailable (Critical!)'}")
    print(f"  - Math Solver: {'Available' if math_solver_available else 'Unavailable'}")
    print(f"  - ML Tools: {'Available' if ml_tools_available else 'Unavailable'}")
    print("-" * 30)

    history = [] # Stores conversation history for the LLM

    while True:
        try:
            prompt = input("You: ")
            if prompt.lower() in ["exit", "quit"]:
                print("AetherMindAI: Goodbye!")
                break

            task_type = detect_task_type(prompt)
            response_str = "" # Renamed from 'response' to avoid conflict
            response_label = "[AetherMind Chat]"

            if task_type == "math":
                if math_solver_available:
                    try:
                        math_result = solve_math_query(prompt)
                        response_str = math_result
                        response_label = "[Math Solver]"
                    except Exception as e:
                        response_str = f"Math solver encountered an error: {e}. Falling back to chat."
                        # Fallback to chat
                        if phi2_available:
                             response_str += "\n[AetherMind Chat]: " + generate_response(phi2_model, phi2_tokenizer, prompt, history)
                        else:
                             response_str += "\n[AetherMind Chat]: Chat model unavailable for fallback."
                        response_label = "[Error/Chat Fallback]"
                else:
                    response_str = "Math solver module is not available. I'll try to answer as a chatbot."
                    if phi2_available:
                        response_str += "\n[AetherMind Chat]: " + generate_response(phi2_model, phi2_tokenizer, prompt, history)
                    else:
                        response_str += "\n[AetherMind Chat]: Chat model unavailable."
                    response_label = "[Chat Fallback]"

            elif task_type == "ml":
                if ml_tools_available:
                    ml_task_name, ml_data_params = get_ml_subtask_and_params(prompt)
                    if ml_task_name and ml_data_params:
                        try:
                            # Ask user if they want a plot for console simplicity, default to False
                            plot_choice = input(f"Run {ml_task_name} task. Include plot (can be verbose)? (yes/no) [no]: ").lower()
                            plot_ml = True if plot_choice == 'yes' else False

                            ml_result = run_ml_task(task_name=ml_task_name, data_params=ml_data_params, plot=plot_ml)
                            response_str = ml_result
                            response_label = f"[ML Tool: {ml_task_name.capitalize()}]"
                        except Exception as e:
                            response_str = f"ML tool encountered an error: {e}. Falling back to chat."
                            if phi2_available:
                                response_str += "\n[AetherMind Chat]: " + generate_response(phi2_model, phi2_tokenizer, prompt, history)
                            else:
                                response_str += "\n[AetherMind Chat]: Chat model unavailable for fallback."
                            response_label = "[Error/Chat Fallback]"
                    else:
                        # If ml_task_name is None, it means it was ambiguous
                        ambiguous_ml_msg = "Could not determine a specific ML task (classification, regression, clustering) from your prompt."
                        response_str = f"{ambiguous_ml_msg} Trying general chat."
                        if phi2_available:
                             response_str += "\n[AetherMind Chat]: " + generate_response(phi2_model, phi2_tokenizer, prompt, history)
                        else:
                             response_str += "\n[AetherMind Chat]: Chat model unavailable."
                        response_label = "[Chat Fallback]"
                else:
                    response_str = "ML tools module is not available. I'll try to answer as a chatbot."
                    if phi2_available:
                        response_str += "\n[AetherMind Chat]: " + generate_response(phi2_model, phi2_tokenizer, prompt, history)
                    else:
                        response_str += "\n[AetherMind Chat]: Chat model unavailable."
                    response_label = "[Chat Fallback]"

            else: # Default to "chat"
                if phi2_available:
                    try:
                        chat_response_text = generate_response(phi2_model, phi2_tokenizer, prompt, history)
                        response_str = chat_response_text
                        # response_label is already "[AetherMind Chat]"
                    except Exception as e:
                        response_str = f"Chat model encountered an error: {e}."
                        response_label = "[Error]"
                else:
                    response_str = "Chat model is not available. Cannot process this request."
                    response_label = "[Error]"

            print(f"{response_label}: {response_str}")

            # Update history: user prompt and assistant response
            history.append({'role': 'user', 'content': prompt})
            # Storing the raw response string without the label for cleaner history for the model
            history.append({'role': 'assistant', 'content': response_str})

            # Keep history to a manageable size (e.g., last 5 interactions = 10 entries)
            if len(history) > 10: # Each interaction is 2 entries (user, assistant)
                history = history[-10:]

        except KeyboardInterrupt:
            print("\nAetherMindAI: Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"An unexpected error occurred in the chat loop: {e}")
            # break # Optional: break loop on unexpected error


if __name__ == "__main__":
    print("\n--- AetherMindAI Controller Initialization Status ---")
    print(f"Phi2 Chat Module: {'Available' if phi2_available and phi2_model and phi2_tokenizer else 'Unavailable'}")
    print(f"Math Solver Module: {'Available' if math_solver_available else 'Unavailable'}")
    print(f"ML Tools Module: {'Available' if ml_tools_available else 'Unavailable'}")
    print("-" * 50)

    if not (phi2_available and phi2_model and phi2_tokenizer):
        print("Core chat module (phi2_colab_runner with model/tokenizer) is not available. AetherMind cannot fully start.")
        # Allow to proceed for testing other modules if they are available
        # sys.exit(1) # Or allow to continue if only some parts are to be tested.

    # --- Test detect_task_type ---
    print("\n--- Testing Task Detection ---")
    test_prompts_detection = {
        "Solve x^2 - 4 = 0": "math",
        "diff(x**2, x)": "math",
        "integrate(sin(x), (x, 0, pi))": "math",
        "Can you run a classification task for me?": "ml",
        "Tell me a joke.": "chat",
        "What is the integral of 2x?": "math", # This could be ambiguous, but current MATH_KEYWORDS will catch it
        "I need to perform clustering on a dataset.": "ml",
        "What's the weather like?": "chat",
        "Calculate the limit of (sin x)/x as x approaches 0": "math",
        "Train a regression model to predict house prices": "ml",
        "what is a math problem?": "math", # Ambiguous, caught by "math problem"
        "run an ml task": "ml"
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
        "Is logistic regression good for this?": ("classification", True),
        "run an ml task for classification": ("classification", True)
    }
    for p_ml, (expected_task, params_expected) in test_ml_prompts.items():
        task, params = get_ml_subtask_and_params(p_ml)
        print(f"Prompt: \"{p_ml}\" | Expected Task: {expected_task} | Detected Task: {task} | Params Expected: {params_expected} | Params Returned: {params is not None}")
        if params:
            print(f"      Params: {params}")

    # --- Ask to run interactive loop ---
    if phi2_available and phi2_model and phi2_tokenizer: # Only offer loop if core chat is working
        print("\n--- Interactive Mode ---")
        run_loop = input("Do you want to start the interactive AetherMindAI chat loop? (yes/no) [yes]: ").lower()
        if run_loop == 'yes' or run_loop == '':
            aethermind_chat_loop()
        else:
            print("Skipping interactive chat loop. End of script.")
    else:
        print("\nSkipping interactive chat loop as core chat module is not available.")
        print("End of script.")
