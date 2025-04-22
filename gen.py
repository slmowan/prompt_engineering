# -*- coding: utf-8 -*-
"""gen.py

Modified from Colab version to run in local environment
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import argparse
import os


class QwenPromptGenerator:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct", device="cuda"):
        """
        Initialize the Qwen model generator.

        Args:
            model_name (str): Hugging Face model identifier for Qwen
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None

        # Load model and tokenizer
        self.load_model()

    def load_model(self):
        """Load the Qwen model and tokenizer"""
        print(f"Loading model: {self.model_name}")

        try:
            # Set the appropriate torch dtype
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if self.device == "cuda" else "cpu",
                trust_remote_code=True
            )

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Configure tokenizer padding settings if needed
            if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print(f"Successfully loaded Qwen model and tokenizer")
            return True

        except Exception as e:
            print(f"Error loading Qwen model: {e}")
            return False

    def generate_response(self, prompt, max_new_tokens=512, temperature=0.7, top_p=0.9, repetition_penalty=1.1):
        """
        Generate a response to the given prompt, handling few-shot Q&A formatting.

        Args:
            prompt (str): Input prompt text
            max_new_tokens (int): Maximum number of new tokens to generate
            temperature (float): Sampling temperature (higher = more creative)
            top_p (float): Nucleus sampling parameter
            repetition_penalty (float): Penalty for repetition

        Returns:
            str: Generated response
        """
        if not self.model or not self.tokenizer:
            print("Model or tokenizer not loaded. Please check initialization.")
            return None

        try:
            # Set up stop sequences for Q/A format
            stop_sequences = ["\n\nQ:", "\nQ:", "\n\n", "\nQuestion:"]

            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,
                    # Some models support stopping sequences directly
                    # But we'll handle this manually in post-processing
                )

            # Decode the generated tokens
            full_response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            # If the full response starts with the prompt, remove the prompt
            if full_response.startswith(prompt):
                response = full_response[len(prompt):].strip()
            else:
                response = full_response.strip()

            # Process the response to extract just the first answer by finding stop sequences
            for stop_seq in stop_sequences:
                if stop_seq in response:
                    response = response.split(stop_seq)[0].strip()
                    break

            # If the response is still long, and it looks like there might be another Q/A pair
            # Use regex to get just the first answer
            import re
            if len(response) > 100 and re.search(r'Q:|Question:', response):
                match = re.match(r'^(.*?)(?=\s*(?:Q:|Question:))', response, re.DOTALL)
                if match:
                    response = match.group(1).strip()

            # Final cleanup
            # If we have a response that starts with "A:" or "Answer:", remove that prefix
            response = re.sub(r'^(?:A:|Answer:)\s*', '', response).strip()

            return response

        except Exception as e:
            print(f"Error generating response: {e}")
            return None


def load_prompts_from_json(json_file):
    """
    Load prompts from a JSON file.

    Args:
        json_file (str): Path to the JSON file containing prompts

    Returns:
        list: List of prompt objects
    """
    try:
        with open(json_file, 'r') as f:
            prompts = json.load(f)
        return prompts
    except Exception as e:
        print(f"Error loading prompts from JSON: {e}")
        return []


def process_prompts(generator, prompts, output_file=None):
    """
    Process all prompts and generate responses.

    Args:
        generator (QwenPromptGenerator): Initialized model generator
        prompts (list): List of prompt objects
        output_file (str): Optional path to save results

    Returns:
        list: List of prompt objects with responses
    """
    results = []

    for i, prompt_obj in enumerate(prompts):
        prompt_id = prompt_obj.get('id', f'prompt_{i}')
        prompt_text = prompt_obj.get('prompt', '')
        category = prompt_obj.get('category', 'unknown')

        print(f"\nProcessing prompt {i + 1}/{len(prompts)}: {prompt_id} (Category: {category})")
        print(f"Prompt: {prompt_text}")

        response = generator.generate_response(prompt_text)

        print(f"Response: {response}")

        # Create result object with prompt and response
        result = {
            "id": prompt_id,
            "category": category,
            "prompt": prompt_text,
            "response": response
        }

        results.append(result)

    # Save results if output file is specified
    if output_file:
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")

    return results


def main():
    # Set up argument parser for local use
    parser = argparse.ArgumentParser(description='Generate responses using Qwen model')
    parser.add_argument('--input', type=str, default="/home/yuming/wanwang/prompt_engineering/data/csci5541-f24-hw5-{NLPeak}-3a.json", help='Input JSON file with prompts')
    parser.add_argument('--output', type=str, default='/home/yuming/wanwang/prompt_engineering/results/output.json', help='Output JSON file for results')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-1.5B-Instruct', help='Model name')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run model on (cuda or cpu)')

    args = parser.parse_args()

    # Initialize the Qwen model
    generator = QwenPromptGenerator(model_name=args.model, device=args.device)

    # Use command line arguments instead of Colab file upload
    prompts_file = args.input
    output_file = args.output

    # Verify the input file exists
    if not os.path.exists(prompts_file):
        print(f"Error: Input file {prompts_file} does not exist.")
        return

    # Load prompts
    prompts = load_prompts_from_json(prompts_file)
    if not prompts:
        print("No prompts loaded. Exiting.")
        return

    print(f"Loaded {len(prompts)} prompts from {prompts_file}")

    # Process all prompts
    results = process_prompts(generator, prompts, output_file)

    print(f"\nProcessed {len(results)} prompts.")


if __name__ == "__main__":
    main()