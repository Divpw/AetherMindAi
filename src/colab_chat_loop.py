def chat_loop(model, tokenizer, max_history_exchanges=4):
    """
    Provides a simple text-based chat interface in Google Colab.

    Args:
        model: The loaded Hugging Face model.
        tokenizer: The loaded Hugging Face tokenizer.
        max_history_exchanges (int): The number of past user/AetherMind exchanges to keep in context.
    """
    chat_history = [] # Stores tuples of (speaker, message)

    print("Starting chat with AetherMind. Type 'exit' or 'quit' to end.")
    print("-" * 30)

    while True:
        try:
            user_input = input("You: ")
        except (KeyboardInterrupt, EOFError): # Handle Ctrl+C or Ctrl+D
            print("\nExiting chat...")
            break

        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat...")
            break

        # Construct the prompt with history
        # For Phi-2, a clear conversational structure is important.
        # Using "Instruct:" and "Output:" might be more for single-turn tasks.
        # For chat, a simple alternation is often a good starting point.
        # Some models respond well to specific role tags if they were trained with them.
        # We'll use a simple "User:" and "AetherMind:" format.

        prompt_parts = []
        for speaker, text in chat_history:
            prompt_parts.append(f"{speaker}: {text}")

        # Add current user input to the prompt for the model
        prompt_parts.append(f"User: {user_input}")
        prompt_parts.append("AetherMind:") # Prompt the model to respond as AetherMind

        full_prompt = "\n".join(prompt_parts)

        # Important: Use the generate_response function defined in phi2_colab_runner.py
        # This assumes generate_response is available in the global scope or imported.
        # For this example, we'll assume it's defined elsewhere.
        # If this code is in the same script/notebook after phi2_colab_runner.py's content, it should work.

        # print(f"\n[DEBUG] Full prompt to model:\n{full_prompt}\n") # For debugging prompt structure

        if 'generate_response' not in globals():
            print("AetherMind: Error - 'generate_response' function is not defined.")
            print("Please ensure 'phi2_colab_runner.py' content with 'generate_response' is executed first.")
            continue # Skip to next user input

        model_response_raw = generate_response(model, tokenizer, full_prompt, max_new_tokens=150) # Adjust max_new_tokens as needed

        # Clean up the response: sometimes models repeat the last part of the prompt or add extra newlines.
        # The generate_response function already tries to strip the prompt if it's at the start.
        # We might also want to strip any leading/trailing whitespace from the actual response part.
        model_response = model_response_raw.strip()

        print(f"AetherMind: {model_response}")

        # Update chat history
        chat_history.append(("User", user_input))
        chat_history.append(("AetherMind", model_response))

        # Keep history to the desired number of exchanges
        # Each exchange has two entries (User, AetherMind)
        if len(chat_history) > max_history_exchanges * 2:
            chat_history = chat_history[-(max_history_exchanges * 2):]

        print("-" * 30)

# Example Usage (assuming model and tokenizer are loaded from phi2_colab_runner.py's functions)
if __name__ == '__main__':
    # This is a placeholder for where you would load your model and tokenizer
    # In a Colab notebook, you would run the cells from phi2_colab_runner.py first.

    print("--- Chat Loop Example ---")
    print("This example requires 'model' and 'tokenizer' to be loaded,")
    print("and the 'generate_response' function to be defined (e.g., from phi2_colab_runner.py).")
    print("Skipping live chat loop in this standalone script execution example.")
    print("To run this interactively, integrate it into your Colab notebook after loading the model.")

    # Mock model and tokenizer for demonstration if generate_response is not available
    # In a real scenario, these would be loaded by load_model_and_tokenizer()
    class MockTokenizer:
        def __init__(self):
            self.pad_token = None
            self.eos_token = "<|endoftext|>"
            self.eos_token_id = 50256 # Example
            self.pad_token_id = 50256 # Example
        def __call__(self, text, return_tensors, return_attention_mask, padding):
            print(f"[MockTokenizer] Called with text: {text[:30]}...")
            class MockInputs:
                def to(self, device): return self
            return MockInputs()
        def decode(self, output, skip_special_tokens):
            return "This is a mock response."

    class MockModelConfig:
        def __init__(self):
            self.pad_token_id = None
            self.eos_token_id = 50256

    class MockModel:
        def __init__(self):
            self.device = "cpu"
            self.config = MockModelConfig()
        def generate(self, **kwargs):
            print(f"[MockModel] Generate called.")
            return [torch.tensor([1,2,3])] # Dummy output

    # Example of how generate_response might be defined if not imported
    if 'generate_response' not in globals():
        print("\nDefining a mock 'generate_response' for testing the chat_loop structure.\n")
        def generate_response(model, tokenizer, prompt, max_new_tokens=50):
            print(f"[Mock generate_response] Prompt: {prompt}")
            # In a real scenario, this calls model.generate() and tokenizer.decode()
            if prompt.strip().endswith("AetherMind:"):
                 return "This is a mock response from AetherMind."
            return "Mock response because the actual model is not generating."

    # To test the chat_loop structure (if you uncomment, run this file directly)
    # print("\n--- Simulating Chat Loop (requires manual input if run) ---")
    # mock_model_instance = MockModel()
    # mock_tokenizer_instance = MockTokenizer()
    # chat_loop(mock_model_instance, mock_tokenizer_instance)
