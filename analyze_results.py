# analyze_results.py
import pandas as pd
import yaml
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_comparison_results(results_file, updated_results_file=None):
    """Analyze and visualize the comparison results"""
    # Load results
    results = pd.read_csv(results_file)

    # If there's an updated file with ChatGPT results, load and merge
    if updated_results_file:
        updated_results = pd.read_csv(updated_results_file)
        results = updated_results

    # Performance Summary
    print("\n=== Performance Summary ===")

    # Model match counts
    local_match_counts = results['Local Model Match'].value_counts().to_dict()
    chatgpt_match_counts = results['ChatGPT Match'].value_counts().to_dict() if 'To be determined' not in results[
        'ChatGPT Match'].unique() else None

    print("\nLocal Model Match Counts:")
    print(local_match_counts)

    if chatgpt_match_counts is not None:
        print("\nChatGPT Match Counts:")
        print(chatgpt_match_counts)

    # Visualization
    plt.figure(figsize=(12, 6))

    # Plot local model results
    plt.subplot(1, 2, 1)
    sns.countplot(x='Local Model Match', data=results)
    plt.title('Local Model Performance')
    plt.xticks(rotation=45)

    # Plot ChatGPT results if available
    if chatgpt_match_counts is not None:
        plt.subplot(1, 2, 2)
        sns.countplot(x='ChatGPT Match', data=results)
        plt.title('ChatGPT Performance')
        plt.xticks(rotation=45)

    plt.tight_layout()

    # Save the figure
    output_dir = os.path.dirname(results_file)
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'))
    plt.close()

    # Detailed prompt-by-prompt comparison
    print("\n=== Detailed Comparison ===")

    # Create JSON structure
    json_results = {
        "prompts": [],
        "summary": {
            "local_model": {
                "status_counts": local_match_counts
            }
        }
    }

    if chatgpt_match_counts is not None:
        json_results["summary"]["chatgpt"] = {
            "status_counts": chatgpt_match_counts
        }

    for i, row in results.iterrows():
        print(f"\nPrompt {i + 1}: {row['Prompt']}")
        print(f"Expected: {row['Expected Output']}")
        print(f"Local Model: {row['Local Model Output']} ({row['Local Model Match']})")
        print(
            f"ChatGPT: {row['ChatGPT Output']} ({row['ChatGPT Match'] if 'To be determined' not in row['ChatGPT Match'] else 'Not evaluated yet'})")

        # Add to JSON structure
        prompt_result = {
            "id": i + 1,
            "prompt": row['Prompt'],
            "expected": row['Expected Output'],
            "local_model": {
                "output": row['Local Model Output'],
                "evaluation": {
                    "match_status": row['Local Model Match']
                }
            },
            "chatgpt": {
                "output": row['ChatGPT Output'],
                "evaluation": {
                    "match_status": row['ChatGPT Match'] if 'To be determined' not in row[
                        'ChatGPT Match'] else "Not evaluated yet"
                }
            }
        }
        json_results["prompts"].append(prompt_result)

    # Save results as JSON
    json_file = os.path.join(output_dir, 'llm_comparison_results.json')
    with open(json_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved as JSON to: {json_file}")

    return results, json_results


def main():
    # Find the most recent results directory
    results_dirs = [d for d in os.listdir() if d.startswith('results_')]
    if not results_dirs:
        print("No results directories found.")
        return

    most_recent_dir = max(results_dirs)
    results_file = os.path.join(most_recent_dir, 'llm_comparison_results.csv')

    # Check if there's an updated file with ChatGPT results
    updated_file = os.path.join(most_recent_dir, 'llm_comparison_results_with_chatgpt.csv')
    if os.path.exists(updated_file):
        results, json_results = analyze_comparison_results(results_file, updated_file)
    else:
        results, json_results = analyze_comparison_results(results_file)

        # Prompt user to update with ChatGPT results
        update = input("\nWould you like to input ChatGPT results now? (y/n): ")
        if update.lower() == 'y':
            for i, row in results.iterrows():
                print(f"\nPrompt: {row['Prompt']}")
                print(f"Expected: {row['Expected Output']}")
                chatgpt_output = input("Enter ChatGPT output: ")

                # Update results
                results.at[i, 'ChatGPT Output'] = chatgpt_output

                # Check match
                if chatgpt_output.lower() == row['Expected Output'].lower():
                    match = "Exact Match"
                elif row['Expected Output'].lower() in chatgpt_output.lower():
                    match = "Partial Match"
                else:
                    match = "No Match"

                results.at[i, 'ChatGPT Match'] = match

            # Save updated results
            results.to_csv(updated_file, index=False)
            print(f"\nUpdated results saved to: {updated_file}")

            # Re-analyze with updated results
            analyze_comparison_results(results_file, updated_file)


if __name__ == "__main__":
    main()
