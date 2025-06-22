import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_model_and_tokenizer(model_id: str = "microsoft/phi-2"):
    """
    Loads a HuggingFace model and tokenizer.
    Attempts to load with 4-bit quantization if GPU is available and bitsandbytes supports it.
    Falls back to loading the model in full precision (float16 or default) on CPU if quantization fails or no GPU.

    Args:
        model_id (str): The HuggingFace model ID (e.g., "microsoft/phi-2").

    Returns:
        tuple: (model, tokenizer) if successful, else (None, None).
    """
    tokenizer = None
    model = None

    # Load tokenizer first
    print(f"Loading tokenizer for '{model_id}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print("Tokenizer loaded successfully.")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("tokenizer.pad_token was None, set to tokenizer.eos_token.")
    except Exception as e:
        print(f"Error loading tokenizer for '{model_id}': {e}")
        return None, None

    # Determine device and quantization strategy
    use_quantization = False
    bnb_config = None
    device_map = "auto" # Default, tries to use GPU if available

    if torch.cuda.is_available():
        try:
            # Check if bitsandbytes can be used with the available CUDA.
            # This is a heuristic check; actual compatibility depends on bitsandbytes compilation.
            # If bitsandbytes itself raises an error during from_pretrained, we'll catch it.
            print("CUDA is available. Attempting to configure 4-bit quantization.")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            use_quantization = True
        except Exception as e: # Catch potential errors from BitsAndBytesConfig itself if CUDA is problematic
            print(f"Could not configure BitsAndBytes for GPU quantization: {e}. Will attempt to load without quantization or on CPU.")
            use_quantization = False
            bnb_config = None
            device_map = "cpu" # Fallback to CPU if quantization setup fails even with CUDA
    else:
        print("CUDA not available. Model will be loaded on CPU, quantization will be skipped.")
        device_map = "cpu" # Explicitly set to CPU

    # Attempt to load model
    try:
        if use_quantization and bnb_config:
            print(f"Loading model '{model_id}' with 4-bit quantization on {device_map}...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                torch_dtype=torch.float16, # Important for compute dtype in quantization
                device_map=device_map,
                trust_remote_code=True,
            )
            print("Model loaded successfully with 4-bit quantization.")
        else:
            print(f"Loading model '{model_id}' without quantization (full precision or float16) on {device_map}...")
            # If falling back, try with float16 for memory, then default if that fails.
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if device_map != "cpu" else None, # float16 on GPU/auto, default on CPU
                    device_map=device_map,
                    trust_remote_code=True,
                )
                print(f"Model loaded successfully in {'float16' if device_map != 'cpu' else 'default'} precision on {device_map}.")
            except Exception as e_fp16: # Fallback to default precision if float16 fails (e.g. on some CPUs)
                print(f"Warning: Failed to load model in float16 on {device_map}: {e_fp16}.")
                print(f"Attempting to load model '{model_id}' in default precision on {device_map}...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map=device_map,
                    trust_remote_code=True,
                )
                print(f"Model loaded successfully in default precision on {device_map}.")

        if hasattr(model, 'device'):
            print(f"Model is effectively on device: {model.device}")

        # Ensure model's config also reflects pad_token_id
        if tokenizer.pad_token == tokenizer.eos_token and \
           (model.config.pad_token_id is None or model.config.pad_token_id != tokenizer.eos_token_id):
            model.config.pad_token_id = tokenizer.eos_token_id
            print(f"Set model.config.pad_token_id to {tokenizer.eos_token_id}")

        return model, tokenizer

    except Exception as e:
        print(f"Error loading model '{model_id}': {e}")
        if "bitsandbytes" in str(e).lower():
            print("This error might be related to bitsandbytes compatibility with your environment (e.g., no GPU, unsupported CUDA version for compiled bnb).")
            print("AetherMind attempted to fall back to CPU loading without quantization, but that also failed.")
        print("Please ensure your environment is set up correctly for the model, or that the model can run on CPU without special requirements.")
        return None, None


def generate_response(model, tokenizer, prompt: str, history: list = None, max_new_tokens: int = 150):
    """
    Generates a response from the model given a prompt and optional history.

    Args:
        model: The pre-loaded HuggingFace model.
        tokenizer: The pre-loaded HuggingFace tokenizer.
        prompt (str): The user's current prompt.
        history (list, optional): A list of dictionaries representing conversation history,
                                  e.g., [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}].
                                  This is for more chat-like interactions if the model supports it.
                                  For Phi-2, a simpler approach might be needed if not using chat templates.
        max_new_tokens (int): Maximum number of new tokens to generate.

    Returns:
        str: The generated response text, or an error message.
    """
    if model is None or tokenizer is None:
        return "Error: Model or tokenizer not loaded. Cannot generate response."

    # For Phi-2, a common way to format input is "Instruct: <instruction>\nOutput:"
    # If history is provided, we can try to build a coherent context.
    # However, Phi-2 is not strictly a chat model, so complex history might not be handled optimally
    # without specific chat templating.

    # Simple history concatenation (adjust if using a specific chat template for Phi-2)
    full_prompt = ""
    if history:
        for turn in history:
            # Assuming 'role' is 'user' or 'assistant' (or similar like 'Instruct'/'Output' for Phi-2)
            # A simple formatting for Phi-2 style (adjust as needed)
            if turn['role'] == 'user':
                full_prompt += f"Instruct: {turn['content']}\nOutput: "
            elif turn['role'] == 'assistant':
                # Append assistant's response and prepare for next instruction
                full_prompt += f"{turn['content']}\n"

    full_prompt += f"Instruct: {prompt}\nOutput:"


    # print(f"\nGenerating response for full_prompt: \"{full_prompt[:200]}...\"") # Debug: print full prompt
    try:
        inputs = tokenizer(full_prompt, return_tensors="pt", return_attention_mask=True, padding=False).to(model.device)

        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_new_tokens": max_new_tokens,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            "do_sample": True,  # Enable sampling
            "temperature": 0.7, # Control randomness; lower is more deterministic
            "top_p": 0.9,       # Nucleus sampling; higher includes more diverse words
            "top_k": 50,        # Consider top_k words at each step
        }

        outputs = model.generate(**generation_kwargs)

        # Decode the entire output first
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the input prompt part from the decoded output
        # The prompt sent to the model was `full_prompt`
        if decoded_output.startswith(full_prompt):
            generated_text = decoded_output[len(full_prompt):].lstrip()
        else:
            # Fallback: if the prompt isn't exactly at the start (e.g., due to tokenization nuances)
            # try to find the last "Output:" and take text after it. This is heuristic.
            last_output_marker = "Output:"
            idx = decoded_output.rfind(last_output_marker)
            if idx != -1:
                generated_text = decoded_output[idx + len(last_output_marker):].lstrip()
            else: # If "Output:" not found, assume the whole non-prompt part is the response
                  # This might happen if the model doesn't follow the Instruct/Output format strictly.
                  # Or, if the generated text is very short.
                  # We still need to remove the original prompt part.
                  # A simple way: decode only the generated tokens (tokens after input_ids.shape[1])
                  input_length = inputs["input_ids"].shape[1]
                  generated_tokens = outputs[0][input_length:]
                  generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).lstrip()


        # print("Response generated successfully.") # Debug
        return generated_text.strip()

    except Exception as e:
        print(f"Error during text generation: {e}")
        return f"Error during generation: {e}"

if __name__ == "__main__":
    print("Starting Phi-2 Colab Runner Test...")

    if torch.cuda.is_available():
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available. Model will run on CPU (slower, quantization might be problematic).")

    # Test loading the model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    if model and tokenizer:
        print("\n--- Model and Tokenizer Loaded Successfully ---")
        print(f"Model class: {model.__class__.__name__}")
        print(f"Tokenizer class: {tokenizer.__class__.__name__}")
        if hasattr(model, 'device'):
            print(f"Model device: {model.device}")
        if tokenizer.pad_token:
            print(f"Tokenizer pad_token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        else:
            print("Tokenizer pad_token is None.")
        print(f"Tokenizer eos_token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")


        # Test generation with a simple prompt
        print("\n--- Test 1: Simple Prompt ---")
        sample_prompt_1 = "What is the capital of France?"
        response_1 = generate_response(model, tokenizer, sample_prompt_1, max_new_tokens=50)
        print(f"Prompt: {sample_prompt_1}")
        print(f"Response: {response_1}")

        # Test generation with a slightly more complex prompt (e.g. few-shot like)
        print("\n--- Test 2: Few-shot Style Prompt ---")
        sample_prompt_2 = "Translate English to French.\nsea otter => loutre de mer\ncheese =>"
        response_2 = generate_response(model, tokenizer, sample_prompt_2, max_new_tokens=10) # Expecting 'fromage'
        print(f"Prompt: {sample_prompt_2}")
        print(f"Response: {response_2}")

        # Test generation with history (simulated)
        print("\n--- Test 3: Prompt with Simulated History ---")
        history_sim = [
            {'role': 'user', 'content': 'My name is Jules.'},
            {'role': 'assistant', 'content': 'Hello Jules! How can I help you today?'}
        ]
        current_prompt = "What is my name?"
        response_3 = generate_response(model, tokenizer, current_prompt, history=history_sim, max_new_tokens=30)
        print(f"History: {history_sim}")
        print(f"Current Prompt: {current_prompt}")
        print(f"Response: {response_3}")

    else:
        print("\nFailed to load model and/or tokenizer. Generation tests skipped.")

    print("\nPhi-2 Colab Runner Test finished.")
