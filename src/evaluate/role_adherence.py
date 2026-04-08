import os, sys
import json
import re
from pathlib import Path


# Import shared components
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]  # …/Trustworthy-LLM-Chatbot-…/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from src.utils.llm import initialize_llm
from src.evaluate.text_process import processed_json_file

def processed_json_file(index, path):
    session_file = f"session_{index}.json"
    with open(os.path.join(path, session_file), "r") as f:
        result = json.load(f)
    dialogue = result["dialogue_history"]

    client_utterances = []
    therapist_utterances = []

    for turn in dialogue:
        if turn['therapist_utterance']:
            therapist_utterances.append(f"Therapist: {turn['therapist_utterance']}")
        if turn['client_utterance']:
            client_utterances.append(f"Client: {turn['client_utterance']}")

    return client_utterances, therapist_utterances

def evaluate_therapist_adherence(therapist_text, rubrics, llm):
    """Evaluate only the therapist's adherence to role"""
    
    # Use only the therapist adherence criteria
    rubric_prompt = f"- Therapist_Adherence ({rubrics['Therapist_Adherence']['question']})"
    format_prompt = "Therapist_Adherence: X"
    
    # Join therapist utterances if it's a list
    if isinstance(therapist_text, list):
        therapist_content = "\n".join(therapist_text)
    else:
        therapist_content = therapist_text
    
    prompt = f"""
    Evaluate the following therapist utterances based on the provided rubric:
    
    {therapist_content}
    
    Score from 1 to 5:
    {rubric_prompt}
    
    Provide score in this format: {format_prompt}
    without any explanation or additional text.
    """
    
    response = llm.invoke(prompt).content
    
    matches = re.findall(r'Therapist_Adherence:\s*(\d+)', response, re.IGNORECASE)
    if matches:
        score = int(matches[0])
        return {"Therapist_Adherence": score}, response
    else:
        return {"Therapist_Adherence": 0}, response

def evaluate_client_adherence(client_text, rubrics, llm):
    """Evaluate only the client's adherence to role"""
    
    # Use only the client adherence criteria
    rubric_prompt = f"- Client_Adherence ({rubrics['Client_Adherence']['question']})"
    format_prompt = "Client_Adherence: X"
    
    # Join client utterances if it's a list
    if isinstance(client_text, list):
        client_content = "\n".join(client_text)
    else:
        client_content = client_text

    prompt = f"""
    Evaluate the following client utterances based on the provided rubric:
    
    {client_content}
    
    Score from 1 to 5:
    {rubric_prompt}
    
    Provide score in this format: {format_prompt}
    without any explanation or additional text.
    """
    
    response = llm.invoke(prompt).content
    
    matches = re.findall(r'Client_Adherence:\s*(\d+)', response, re.IGNORECASE)
    if matches:
        score = int(matches[0])
        return {"Client_Adherence": score}, response
    else:
        return {"Client_Adherence": 0}, response

def save_evaluation_results(result, output_path, model_name):
    """Save the evaluation results to a JSON file."""
    output_file = output_path / f"role_adherence_{model_name}.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=4)
    print(f"Results saved to {output_file}")

def main():
    # Load the rubrics
    with open(PROJECT_ROOT / "src/evaluate/evaluation_rubrics.json", "r") as f:
        rubrics = json.load(f)

    # Initialize the LLM
    llm = initialize_llm(local_llm=True, model_name="llama3.1:8b", temperature=1, n=10, top_p=1)

    model_name = "phi4:14b"
    eval_path = PROJECT_ROOT / f"data/results/sessions/level1_by2llm/{model_name}"
    all_results = {}
    for index in range(284, 286):
        client_utterances, therapist_utterances = processed_json_file(index, eval_path)
    
        # Evaluate therapist and client separately
        therapist_scores, therapist_response = evaluate_therapist_adherence(therapist_utterances, rubrics, llm)
        client_scores, client_response = evaluate_client_adherence(client_utterances, rubrics, llm)
        
        print("Therapist Role Adherence Score:", therapist_scores)
        print("Client Role Adherence Score:", client_scores)
        
        # Combine scores for overall results
        combined_scores = {**therapist_scores, **client_scores}
        print(f"Combined Scores for Session {index}: {combined_scores}")
        
        all_results[index] = combined_scores
    
    # Save all results to a JSON file
    output_path = PROJECT_ROOT / "data/results/evaluation_results"
    output_path.mkdir(parents=True, exist_ok=True)
    save_evaluation_results(all_results, output_path)
    return all_results

if __name__ == "__main__":
    results = main()
    print("Final Results:", results)