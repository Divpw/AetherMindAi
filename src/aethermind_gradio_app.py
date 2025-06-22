# aethermind_gradio_app.py
# This file will contain the Gradio interface for AetherMindAI.

import gradio as gr
import sys
import os

# Adjust system path to include parent directory for src and reasoning_engine imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) # Assuming this script is in a subdir like 'apps' or 'ui'
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path: # If script is in root, current_dir might be parent_dir
    sys.path.insert(0, current_dir)


# Attempt to import backend functions
try:
    from reasoning_engine import run_reasoning_prompt
except ImportError:
    print("Warning: reasoning_engine.run_reasoning_prompt not found. Using placeholder.")
    def run_reasoning_prompt(prompt: str) -> str:
        return f"Placeholder for Reasoning: Received '{prompt}'. Implement run_reasoning_prompt."

try:
    from src.colab_math_solver import solve_math_query
except ImportError:
    print("Warning: src.colab_math_solver.solve_math_query not found. Using placeholder.")
    def solve_math_query(query: str) -> str:
        return f"Placeholder for Math: Received '{query}'. Implement solve_math_query."

try:
    from src.colab_ml_tools import run_ml_task
except ImportError:
    print("Warning: src.colab_ml_tools.run_ml_task not found. Using placeholder.")
    def run_ml_task(task_name: str, data_params: dict, model_params: dict = None, plot: bool = False):
        # This placeholder needs to return a string and potentially a plot object if Gradio expects it
        response_str = f"Placeholder for ML: Task '{task_name}' with data_params {data_params}, plot={plot}. Implement run_ml_task."
        if plot:
            # If run_ml_task is expected to return a plot object for gr.Plot
            # For now, let's assume it either displays it directly (plt.show()) or returns None for plot if placeholder
            # Or, we can return a dummy plot object if needed by Gradio output components
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, "Placeholder Plot", ha='center', va='center')
                plt.close(fig) # Close the figure to prevent auto-display by matplotlib here
                return response_str, fig
            except ImportError:
                return response_str, None # No plot if matplotlib not available
        return response_str, None # Return None for plot if plot=False

# For Chat mode
CHAT_MODEL = None
CHAT_TOKENIZER = None
CHAT_MODEL_LOADED = False

try:
    from src.phi2_colab_runner import load_model_and_tokenizer, generate_response as phi2_generate_response
    # Load model and tokenizer for Chat mode globally - this might take time.
    # Consider lazy loading or loading indicator in a real app.
    print("Loading Chat model and tokenizer... This may take a while.")
    CHAT_MODEL, CHAT_TOKENIZER = load_model_and_tokenizer() # Using default "microsoft/phi-2"
    if CHAT_MODEL and CHAT_TOKENIZER:
        CHAT_MODEL_LOADED = True
        print("Chat model and tokenizer loaded successfully.")
    else:
        print("Failed to load Chat model and tokenizer. Chat mode will use a placeholder.")
except ImportError:
    print("Warning: src.phi2_colab_runner not found. Chat mode will use a placeholder.")
    phi2_generate_response = None # Ensure it's defined for the placeholder below

if not CHAT_MODEL_LOADED:
    # Placeholder for generate_response if model loading failed or phi2_colab_runner not found
    def generate_chat_response_placeholder(prompt: str, history: list = None): # Matched to Gradio call
        # The Gradio app will pass (prompt_input, chat_history_state)
        # The actual phi2_generate_response takes (model, tokenizer, prompt, history, max_new_tokens)
        return f"Placeholder for Chat: Model not loaded. Received '{prompt}'. History: {history}"
else:
    # Wrapper for the actual generate_response to match what Gradio might send simply
    def generate_chat_response_wrapper(prompt: str, history: list = None): # Matched to Gradio call
        # The Gradio app will pass (prompt_input, chat_history_state)
        # The actual phi2_generate_response takes (model, tokenizer, prompt, history, max_new_tokens)
        # We need to adapt. 'history' from Gradio chatbot is usually a list of lists [[user_msg, bot_msg], ...]
        # phi2_generate_response expects history as [{'role': 'user', 'content': '...'}, ...]

        formatted_history = []
        if history:
            for user_msg, bot_msg in history:
                if user_msg: formatted_history.append({'role': 'user', 'content': user_msg})
                if bot_msg: formatted_history.append({'role': 'assistant', 'content': bot_msg})

        # Call the actual function from phi2_colab_runner
        return phi2_generate_response(CHAT_MODEL, CHAT_TOKENIZER, prompt, history=formatted_history)


# Main launch function
def launch_gradio_interface():
    """
    Defines and launches the Gradio interface for AetherMindAI.
    """
    # Determine which chat function to use
    active_chat_handler = generate_chat_response_wrapper if CHAT_MODEL_LOADED else generate_chat_response_placeholder

    # ML Task specific configurations
    ml_task_choices = ["classification", "regression", "clustering"]
    default_ml_data_params_str = """{
    "n_samples": 100,
    "n_features": 2,
    "random_state": 42
}"""

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="light_blue"), title="AetherMindAI Assistant") as aethermind_iface:
        gr.Markdown("# AetherMindAI Assistant")

        with gr.Row():
            with gr.Column(scale=1, min_width=200): # Sidebar/Mode selection
                mode = gr.Radio(
                    ["Chat", "Math", "ML", "Reasoning"],
                    label="Select Mode",
                    value="Chat"
                )

                # ML Specific Controls (conditionally visible)
                ml_task_dropdown = gr.Dropdown(
                    ml_task_choices,
                    label="ML Task Type",
                    value="classification",
                    visible=False # Initially hidden
                )
                ml_data_params_textbox = gr.Textbox(
                    label="Data Parameters (JSON string)",
                    value=default_ml_data_params_str,
                    lines=4,
                    visible=False # Initially hidden
                )
                ml_plot_checkbox = gr.Checkbox(
                    label="Show Plot",
                    value=True,
                    visible=False # Initially hidden
                )

            with gr.Column(scale=4): # Main interaction area
                # Chatbot component for Chat mode
                chatbot_display = gr.Chatbot(
                    label="AetherMind Chat",
                    bubble_full_width=False,
                    height=500,
                    visible=True # Visible by default for Chat mode
                )
                # Textbox for other modes' output
                text_output_display = gr.Textbox(
                    label="Output",
                    lines=10,
                    interactive=False,
                    visible=False # Hidden by default
                )
                # Plot for ML mode
                plot_output_display = gr.Plot(
                    label="ML Plot Output",
                    visible=False # Hidden by default
                )

                prompt_input = gr.Textbox(
                    label="Your Input / Prompt / Query",
                    placeholder="Type your message here...",
                    lines=3
                )
                submit_button = gr.Button("Send / Execute", variant="primary")

        # Chat history state for Chat mode
        chat_history_state = gr.State([])

        # --- UI Logic and Interactions ---

        def handle_mode_change(selected_mode):
            """Updates visibility of UI elements based on selected mode."""
            is_chat_mode = selected_mode == "Chat"
            is_ml_mode = selected_mode == "ML"

            # Output displays
            chatbot_visible = is_chat_mode
            text_output_visible = not is_chat_mode # Math, Reasoning, ML text output
            plot_output_visible = is_ml_mode

            # ML specific controls
            ml_controls_visible = is_ml_mode

            return {
                # Output components
                chatbot_display: gr.update(visible=chatbot_visible),
                text_output_display: gr.update(visible=text_output_visible, value=""), # Clear previous output
                plot_output_display: gr.update(visible=plot_output_visible, value=None), # Clear previous plot
                # ML Controls
                ml_task_dropdown: gr.update(visible=ml_controls_visible),
                ml_data_params_textbox: gr.update(visible=ml_controls_visible),
                ml_plot_checkbox: gr.update(visible=ml_controls_visible),
                # Prompt input placeholder
                prompt_input: gr.update(placeholder= "Type your message here..." if is_chat_mode \
                                        else "Enter your query or command..." if selected_mode == "Math" \
                                        else "Enter prompt for ML or Reasoning..." )
            }

        mode.change(
            fn=handle_mode_change,
            inputs=mode,
            outputs=[
                chatbot_display, text_output_display, plot_output_display,
                ml_task_dropdown, ml_data_params_textbox, ml_plot_checkbox,
                prompt_input
            ]
        )

        def handle_submit(current_mode, user_prompt, ch_history, ml_task, ml_data_params_str, ml_plot):
            """Master submit handler. Routes to appropriate backend function based on mode."""
            ch_history = ch_history or []
            text_result = ""
            plot_result = None # For ML plots

            if not user_prompt and current_mode != "ML": # ML might run with default params without prompt
                 if current_mode == "Chat": # For chat, don't submit empty, just clear prompt
                    return "", ch_history, None, None # prompt, history, text_out, plot_out
                 else: # For other modes, show error if prompt is empty
                    return user_prompt, ch_history, "Error: Input prompt/query cannot be empty.", None


            if current_mode == "Chat":
                bot_message = active_chat_handler(user_prompt, ch_history)
                ch_history.append((user_prompt, bot_message))
                return "", ch_history, None, None # Clear prompt, update history, no text/plot output for these components

            elif current_mode == "Math":
                text_result = solve_math_query(user_prompt)

            elif current_mode == "Reasoning":
                text_result = run_reasoning_prompt(user_prompt)

            elif current_mode == "ML":
                try:
                    import json
                    data_params = json.loads(ml_data_params_str)
                    # Note: run_ml_task currently returns (str_summary, fig_object_or_None_if_placeholder)
                    # If run_ml_task uses plt.show() and doesn't return a fig, gr.Plot won't auto-update.
                    # For now, assuming run_ml_task is modified or placeholder returns a figure.
                    # The placeholder for run_ml_task returns (str, fig or None).
                    # The actual run_ml_task in colab_ml_tools.py uses plt.show() and returns a string.
                    # This needs reconciliation. For now, we assume it returns string and figure.
                    # TODO: Adjust if run_ml_task does not return a figure object.

                    # Let's call the actual run_ml_task which returns a string and internally calls plt.show()
                    # Gradio's gr.Plot() can sometimes capture plt.show() if it's the last thing.
                    # However, to be explicit, it's better if run_ml_task returns the figure.
                    # For this iteration, we'll use the placeholder's behavior for return type.

                    # Using the imported run_ml_task, which might be a placeholder or the real one.
                    # The placeholder returns (str, fig_or_None).
                    # The real one from colab_ml_tools.py returns str and uses plt.show().
                    ml_output = run_ml_task(ml_task, data_params, plot=ml_plot)
                    if isinstance(ml_output, tuple) and len(ml_output) == 2:
                        text_result, plot_result = ml_output # Placeholder or modified real function
                    else:
                        text_result = ml_output # Original real function from colab_ml_tools
                        plot_result = None # No figure returned by original function
                        if ml_plot: # Add a note if user expected a plot in the Gradio component
                            text_result += "\n\nNote: ML task executed. Plot (if any) was displayed by the backend via plt.show(). For direct display in Gradio plot area, the backend would need to return a Matplotlib Figure object."

                except json.JSONDecodeError as e:
                    text_result = f"Error parsing Data Parameters JSON: {e}"
                    plot_result = None
                except Exception as e:
                    text_result = f"Error in ML task execution: {e}"
                    plot_result = None

            # For non-chat modes, prompt is not cleared automatically unless we return "" for prompt_input
            # For now, let's keep the prompt input as is for Math, ML, Reasoning after submit
            return user_prompt, ch_history, text_result, plot_result


        submit_button.click(
            fn=handle_submit,
            inputs=[mode, prompt_input, chat_history_state, ml_task_dropdown, ml_data_params_textbox, ml_plot_checkbox],
            outputs=[prompt_input, chat_history_state, text_output_display, plot_output_display]
        )

        # Allow submitting prompt with Enter key
        prompt_input.submit(
            fn=handle_submit,
            inputs=[mode, prompt_input, chat_history_state, ml_task_dropdown, ml_data_params_textbox, ml_plot_checkbox],
            outputs=[prompt_input, chat_history_state, text_output_display, plot_output_display]
        )


    print("Gradio UI definition complete. Launching interface...")
    aethermind_iface.launch()


if __name__ == '__main__':
    print("AetherMind Gradio App - Attempting to load components and launch UI...")
    launch_gradio_interface()
    # Test calls to ensure placeholders or real functions are picked up (these will print to console before UI launches)
    print("\nTesting imported/placeholder functions (output will appear in console before UI starts):")
    print(f"Reasoning: {run_reasoning_prompt('Test reasoning')}")
    print(f"Math: {solve_math_query('2+2')}")

    # Handle potential tuple or string return from run_ml_task (placeholder vs real)
    ml_output = run_ml_task('classification', {'n_samples':50, 'random_state':42}, plot=False)
    if isinstance(ml_output, tuple):
        print(f"ML (no plot): {ml_output[0]}") # Placeholder returns (str, fig_or_None)
    else:
        print(f"ML (no plot): {ml_output}") # Real function returns str

    if CHAT_MODEL_LOADED:
        # This is a direct call, Gradio's chat history state isn't involved here
        print(f"Chat (loaded - direct call): {generate_chat_response_wrapper('Hello AetherMind', [])}")
    else:
        print(f"Chat (placeholder - direct call): {generate_chat_response_placeholder('Hello AetherMind', [])}")

    print("\nThe Gradio interface should be launching in your browser or inline if in a notebook.")
    print("If the UI has launched, the test prints above have already executed.")
