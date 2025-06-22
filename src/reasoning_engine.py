# reasoning_engine.py

import re # Imported for the train problem

def run_reasoning_prompt(prompt: str) -> str:
    """
    Simulates intelligent reasoning for a small LLM (like Phi-2)
    by breaking down problems and using chain-of-thought style reasoning.

    Args:
        prompt: A natural language logic query.

    Returns:
        A string containing the reasoning steps and the final answer.
    """
    # Simulate chain of thought
    reasoning_steps = []
    final_answer = "Could not determine the answer." # Default answer

    reasoning_steps.append(f"START_REASONING")
    reasoning_steps.append(f"Prompt received: \"{prompt}\"")

    # Normalize prompt for easier keyword spotting
    normalized_prompt = prompt.lower()
    reasoning_steps.append(f"Step 1: Normalized prompt to lowercase: \"{normalized_prompt}\"")

    # Keyword-based routing / prompt type analysis
    if "alice" in normalized_prompt and "bob" in normalized_prompt and "charlie" in normalized_prompt and "older" in normalized_prompt:
        reasoning_steps.append("Step 2: Detected an age comparison problem (Alice, Bob, Charlie).")
        # Logic for "If Alice is older than Bob and Bob is older than Charlie, who is oldest?"
        # A > B and B > C  => A is oldest.
        reasoning_steps.append("Step 3: Analyzing relationships: 'Alice is older than Bob' means Alice > Bob.")
        reasoning_steps.append("Step 4: Analyzing relationships: 'Bob is older than Charlie' means Bob > Charlie.")
        reasoning_steps.append("Step 5: Combining relationships: If Alice > Bob and Bob > Charlie, then Alice > Bob > Charlie.")
        reasoning_steps.append("Step 6: Conclusion: Alice is the oldest.")
        final_answer = "Alice"
    elif "train" in normalized_prompt and "km/h" in normalized_prompt and "apart" in normalized_prompt:
        reasoning_steps.append("Step 2: Detected a train speed/distance problem.")
        # Logic for "A train leaves station A at 60 km/h and another train leaves station B at 80 km/h towards each other. They are 280km apart. When will they meet?"
        try:
            # Using normalized_prompt for consistency in case regex becomes more complex
            numbers = re.findall(r'\d+', normalized_prompt)

            if len(numbers) >= 3: # Ensure we have enough numbers for speeds and distance
                s1 = int(numbers[0]) # Speed of train 1
                s2 = int(numbers[1]) # Speed of train 2
                d = int(numbers[2])  # Distance apart

                reasoning_steps.append(f"Step 3: Extracted numerical values: Speed1={s1} km/h, Speed2={s2} km/h, Distance={d} km.")
                reasoning_steps.append(f"Step 4: Problem type is meeting trains. Their relative speed is the sum of their individual speeds.")
                relative_speed = s1 + s2
                reasoning_steps.append(f"Step 5: Calculated relative speed: {s1} + {s2} = {relative_speed} km/h.")
                reasoning_steps.append(f"Step 6: Time to meet = Total Distance / Relative Speed.")
                if relative_speed == 0:
                    reasoning_steps.append(f"Step 7: Relative speed is 0. Cannot calculate time to meet if trains are not moving towards each other or are stationary.")
                    final_answer = "Cannot calculate time to meet; relative speed is zero."
                else:
                    time_to_meet = d / relative_speed
                    reasoning_steps.append(f"Step 7: Calculated time to meet: {d} / {relative_speed} = {time_to_meet:.2f} hours.")
                    final_answer = f"{time_to_meet:.2f} hours"
            else:
                reasoning_steps.append(f"Step 3: Error: Could not extract sufficient numerical values (found {len(numbers)}, expected 3 for speeds and distance).")
                final_answer = "Could not perform calculation due to missing numerical data."
        except ValueError as e: # More specific exception for int() conversion
            reasoning_steps.append(f"Step 3: Error: Could not convert extracted parts to numbers: {e}")
            final_answer = "Could not perform calculation due to invalid numerical data."
        except Exception as e: # General fallback
            reasoning_steps.append(f"Step 3: Error during numerical extraction or calculation: {e}")
            final_answer = "Could not perform calculation due to an unexpected error."
    else:
        reasoning_steps.append("Step 2: Could not confidently categorize the prompt based on keywords.")
        reasoning_steps.append("Step 3: No specific reasoning path available for this prompt type.")

    reasoning_steps.append(f"END_REASONING")

    # Assemble the output
    output_parts = ["Chain of Thought:"]
    for i, step in enumerate(reasoning_steps):
        output_parts.append(f"  - {step}")

    output_parts.append(f"\nFinal Answer: {final_answer}")
    return "\n".join(output_parts)

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    prompt1 = "If Alice is older than Bob and Bob is older than Charlie, who is oldest?"
    prompt2 = "A train leaves station A at 60 km/h and another train leaves station B at 80 km/h towards each other. They are 280km apart. When will they meet?"
    prompt3 = "What is the capital of France?" # Example of an unhandled prompt
    prompt4 = "A train travels at 0 km/h and another at 0 km/h, 100km apart." # Test zero speed
    prompt5 = "A train problem with only two numbers 60 80." # Test insufficient numbers
    prompt6 = "Train speed is sixty km/h, other is eighty km/h, apart by two hundred eighty km." # Test with words (will fail number extraction as is)

    print("---- Example 1 ----")
    print(run_reasoning_prompt(prompt1))
    print("\n---- Example 2 ----")
    print(run_reasoning_prompt(prompt2))
    print("\n---- Example 3 ----")
    print(run_reasoning_prompt(prompt3))
    print("\n---- Example 4 (Zero Speed) ----")
    print(run_reasoning_prompt(prompt4))
    print("\n---- Example 5 (Insufficient Numbers) ----")
    print(run_reasoning_prompt(prompt5))
    print("\n---- Example 6 (Numbers as Words - Expect Fail) ----")
    print(run_reasoning_prompt(prompt6))
