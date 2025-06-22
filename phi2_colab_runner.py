import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_model_and_tokenizer():
    """
    Loads the Phi-2 model and tokenizer with 4-bit quantization.
    """
    model_id = "microsoft/phi-2"

    # Configuration for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True, # Optional: can lead to slightly more memory saving
    )

    print(f"Loading tokenizer for '{model_id}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print("Tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None, None

    print(f"Loading model '{model_id}' with 4-bit quantization...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            torch_dtype=torch.float16, # Explicitly set, though compute_dtype is in bnb_config
            device_map="auto", # Automatically uses GPU if available
            trust_remote_code=True,
        )
        print("Model loaded successfully.")
        if model.device.type != 'cpu':
            print(f"Model is on device: {model.device} (GPU)")
        else:
            print(f"Model is on device: {model.device} (CPU)")

        # Set pad_token if not set, common for some models like Phi-2
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
            print("tokenizer.pad_token was None, set to tokenizer.eos_token.")

        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure you have a GPU available and enough VRAM. For Colab, ensure T4 GPU is selected.")
        print("You might also need to accept Hugging Face terms for Phi-2 if you haven't.")
        return None, None

# Script content will be added in subsequent steps.
# For testing this function:
# if __name__ == '__main__':
# model, tokenizer = load_model_and_tokenizer()
# if model and tokenizer:
# print("Test: Model and tokenizer loaded.")
# else:
# print("Test: Failed to load model or tokenizer.")

def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 100):
    """
    Generates a response from the model given a prompt.
    """
    if model is None or tokenizer is None:
        print("Model or tokenizer not available. Cannot generate response.")
        return "Error: Model or tokenizer not loaded."

    # Double check pad_token (should be handled in load_model_and_tokenizer, but good practice)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # Ensure model's config also reflects this, if necessary for generation
        if model.config.pad_token_id is None:
             model.config.pad_token_id = model.config.eos_token_id


    print(f"\nGenerating response for prompt: \"{prompt[:100]}...\"")
    try:
        # Ensure inputs are on the same device as the model
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True, padding=True).to(model.device)

        # Generate text
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id, # Important for some models/padding strategies
            # num_beams=5, # Optional: for beam search
            # early_stopping=True, # Optional
            # no_repeat_ngram_size=2 # Optional
        )

        # Decode the output, skipping special tokens like <|endoftext|>
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Phi-2 (and some other models) might return the prompt as part of the output.
        # We can remove it if present.
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].lstrip()

        print("Response generated successfully.")
        return generated_text
    except Exception as e:
        print(f"Error during text generation: {e}")
        return f"Error during generation: {e}"

if __name__ == "__main__":
    print("Starting Phi-2 Colab Runner...")

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("GPU not available. Model will run on CPU (slower and might not work with quantization).")

    model, tokenizer = load_model_and_tokenizer()

    if model and tokenizer:
        sample_prompt = "What is the capital of France?"
        # For Phi-2, specific instruction or question formatting might be better.
        # E.g., "Instruct: What is the capital of France?\nOutput:"
        # Or for chat models: tokenizer.apply_chat_template([{"role": "user", "content": sample_prompt}], tokenize=False)

        print(f"\nUsing prompt: \"{sample_prompt}\"")

        response = generate_response(model, tokenizer, sample_prompt, max_new_tokens=50)

        print("\n--- Prompt ---")
        print(sample_prompt)
        print("\n--- Response ---")
        print(response)
    else:
        print("\nFailed to load model and/or tokenizer. Exiting.")

    print("\nPhi-2 Colab Runner finished.")
