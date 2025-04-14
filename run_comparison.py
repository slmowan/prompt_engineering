# run_comparison.py
import os
import yaml
import json
import torch
from datetime import datetime
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
# from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login


class LLMComparisonPipeline:
    def __init__(self, config_file):
        # Load configuration
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)

        # Create output directory for results
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"./results/results_{self.timestamp}"
        os.makedirs("./results", exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize models
        self.local_llm = None
        self.local_chain = None

    def setup_local_model(self):
        """Set up the local model using LangChain following the tutorial approach"""
        model_config = self.config['models']['local']
        model_name = model_config['name']
        params = model_config['parameters']

        print(f"Loading model: {model_name}")

        try:
            # Convert string dtype to torch dtype
            if params['torch_dtype'] == 'float16':
                torch_dtype = torch.float16
            elif params['torch_dtype'] == 'bfloat16':
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32

            # Load model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=params['device_map'],
                trust_remote_code=True
            )

            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Configure tokenizer padding settings if needed
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token

            # Create HuggingFace pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=params['max_new_tokens'],
                temperature=params['temperature'],
                top_p=params['top_p'],
                repetition_penalty=params['repetition_penalty']
            )

            # Create LangChain LLM
            self.local_llm = HuggingFacePipeline(pipeline=pipe)

            # Create LangChain prompt template
            prompt_template = self.config.get('prompt_template', "{input}")
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["input"]
            )

            # Create LangChain chain
            self.local_chain = LLMChain(llm=self.local_llm, prompt=prompt)
            # from langchain.schema.runnable import RunnablePassthrough
            # self.local_chain = prompt | self.local_llm

            print(f"Successfully loaded model and created LangChain chain")
            return self.local_chain

        except Exception as e:
            print(f"Error setting up local model: {e}")
            return None

    def run_comparison(self):
        """Run the comparison using LangChain"""
        # Set up the model if not already done
        if not self.local_chain:
            self.setup_local_model()
            if not self.local_chain:
                print("Failed to set up the model. Exiting.")
                return None

        # Get prompt pairs
        prompt_pairs = self.config['prompt_pairs']
        results = {"prompts": [], "summary": {}}

        # Track successes and failures
        status_counts = {"Exact Match": 0, "Partial Match": 0, "No Match": 0, "Error": 0}

        # Process each prompt
        for i, pair in enumerate(prompt_pairs):
            input_text = pair['prompt']
            expected = pair['expected']

            print(f"\nProcessing prompt {i + 1}/{len(prompt_pairs)}: {input_text}")

            # Run the prompt through the LangChain chain
            try:
                # Use the LangChain chain to generate a response
                response = self.local_chain.invoke({"input": input_text})
                output = response.get("text", "").strip()
                error = None
                print(f"Local model output: {output}")
            except Exception as e:
                output = None
                error = str(e)
                print(f"Error getting response: {error}")

            # Evaluate the response
            if error:
                match_status = "Error"
                match_score = 0.0
                status_counts["Error"] += 1
            elif output is None:
                match_status = "Error"
                match_score = 0.0
                status_counts["Error"] += 1
            elif output.lower() == expected.lower():
                match_status = "Exact Match"
                match_score = 1.0
                status_counts["Exact Match"] += 1
            elif expected.lower() in output.lower():
                match_status = "Partial Match"
                match_score = 0.5
                status_counts["Partial Match"] += 1
            else:
                match_status = "No Match"
                match_score = 0.0
                status_counts["No Match"] += 1

            # Build result object
            prompt_result = {
                "id": i + 1,
                "prompt": input_text,
                "expected": expected,
                "local_model": {
                    "name": self.config['models']['local']['name'],
                    "output": output,
                    "error": error,
                    "evaluation": {
                        "match_status": match_status,
                        "match_score": match_score
                    }
                },
                "chatgpt": {
                    "name": self.config['models']['chatgpt']['model'],
                    "prompt_for_manual_testing": input_text,
                    "output": "To be filled manually",
                    "evaluation": {
                        "match_status": "Not evaluated yet",
                        "match_score": None
                    }
                }
            }

            results["prompts"].append(prompt_result)

        # Add summary to results
        results["summary"] = {
            "local_model": {
                "name": self.config['models']['local']['name'],
                "status_counts": status_counts,
                "average_match_score": sum(
                    prompt["local_model"]["evaluation"]["match_score"] for prompt in results["prompts"]) / len(
                    results["prompts"]) if results["prompts"] else 0
            },
            "chatgpt": {
                "name": self.config['models']['chatgpt']['model'],
                "status": "Pending manual evaluation"
            }
        }

        # Save results as JSON
        results_file = os.path.join(self.output_dir, 'llm_comparison_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Save configuration for reproducibility
        with open(os.path.join(self.output_dir, 'config_used.yaml'), 'w') as f:
            yaml.dump(self.config, f)

        print(f"\nResults saved to: {results_file}")

        return results

    def generate_chatgpt_instructions(self):
        """Generate instructions for manual ChatGPT testing"""
        # Load results to get the formatted prompts
        results_file = os.path.join(self.output_dir, 'llm_comparison_results.json')
        with open(results_file, 'r') as f:
            results = json.load(f)

        instructions = "# ChatGPT Testing Instructions\n\n"
        instructions += "Copy and paste each of the following prompts into ChatGPT and record the responses:\n\n"

        for prompt_result in results["prompts"]:
            instructions += f"## Prompt {prompt_result['id']}:\n```\n{prompt_result['prompt']}\n```\n\n"
            instructions += f"Expected output: {prompt_result['expected']}\n\n"
            instructions += "ChatGPT response: [Copy response here]\n\n"
            instructions += "---\n\n"

        instructions_file = os.path.join(self.output_dir, 'chatgpt_instructions.md')
        with open(instructions_file, 'w') as f:
            f.write(instructions)

        print(f"ChatGPT testing instructions saved to: {instructions_file}")

    def update_with_chatgpt_results(self, chatgpt_results):
        """Update the results with manually collected ChatGPT responses"""
        # Load existing results
        results_file = os.path.join(self.output_dir, 'llm_comparison_results.json')
        with open(results_file, 'r') as f:
            results = json.load(f)

        # Update with ChatGPT results
        chatgpt_status_counts = {"Exact Match": 0, "Partial Match": 0, "No Match": 0}
        total_score = 0

        for i, response in enumerate(chatgpt_results):
            if i < len(results["prompts"]):
                expected = results["prompts"][i]["expected"]

                # Evaluate ChatGPT response
                if response.lower() == expected.lower():
                    match_status = "Exact Match"
                    match_score = 1.0
                    chatgpt_status_counts["Exact Match"] += 1
                elif expected.lower() in response.lower():
                    match_status = "Partial Match"
                    match_score = 0.5
                    chatgpt_status_counts["Partial Match"] += 1
                else:
                    match_status = "No Match"
                    match_score = 0.0
                    chatgpt_status_counts["No Match"] += 1

                total_score += match_score

                # Update the result
                results["prompts"][i]["chatgpt"]["output"] = response
                results["prompts"][i]["chatgpt"]["evaluation"] = {
                    "match_status": match_status,
                    "match_score": match_score
                }

        # Update summary
        if chatgpt_results:
            results["summary"]["chatgpt"] = {
                "name": self.config['models']['chatgpt']['model'],
                "status_counts": chatgpt_status_counts,
                "average_match_score": total_score / len(chatgpt_results) if chatgpt_results else 0
            }

        # Save updated results
        updated_file = os.path.join(self.output_dir, 'llm_comparison_results_final.json')
        with open(updated_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Updated results saved to: {updated_file}")

        return results


def main():
    # Check for HuggingFace token
    if os.environ.get('HUGGINGFACE_TOKEN') is None:
        token = input("Enter your HuggingFace token (or press Enter to skip): ")
        if token:
            try:
                login(token=token)
                print("Successfully logged in to Hugging Face")
            except Exception as e:
                print(f"Error logging in: {e}")
                print("You may need to accept model licenses on the Hugging Face website.")
        else:
            print("No token provided. Some models may not be accessible.")
    else:
        try:
            login(token=os.environ['HUGGINGFACE_TOKEN'])
            print("Successfully logged in to Hugging Face")
        except Exception as e:
            print(f"Error logging in: {e}")

    # Create and run pipeline
    pipeline = LLMComparisonPipeline('configs.yaml')
    results = pipeline.run_comparison()

    if results:
        # Generate ChatGPT testing instructions
        pipeline.generate_chatgpt_instructions()

        # Print quick summary
        if "summary" in results and "local_model" in results["summary"]:
            print("\nLocal model performance summary:")
            for status, count in results["summary"]["local_model"]["status_counts"].items():
                print(f"  {status}: {count}")

        print(
            "\nPlease follow the instructions in the 'chatgpt_instructions.md' file to complete the ChatGPT portion of the comparison.")
        print("After collecting ChatGPT responses, run the update_chatgpt_results.py script to finalize your results.")


if __name__ == "__main__":
    main()
