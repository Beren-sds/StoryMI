"""
Text processing utility module containing functions for handling dialogue data.
"""
import pandas as pd
import os
import json
import pprint
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent

# Data path configuration
GENERATED_DIALOGUE_PATH = os.path.join(ROOT_DIR, "data/results/sessions/level1_by2llm/llama3.1:8b")
REFERENCE_DIALOGUE_PATH = os.path.join(ROOT_DIR, "data/midatasets/AnnoMI-full.csv")

def processed_json_file(index, path):
    """
    Process generated JSON dialogue files
    Args:
        num (int): File index
    Returns:
        tuple: (Formatted dialogue list, intent label dictionary)
    """
    # session_files = [f for f in os.listdir(path) if f.endswith('.json')]
    # sorted_files = sorted(session_files, key=lambda x: int(''.join(filter(str.isdigit, x))))
    session_file = f"session_{index}.json"
    with open(os.path.join(path, session_file), "r") as f:
        result = json.load(f)
    dialogue = result["dialogue_history"]

    formatted_dialogue = []
    intent_label = {
        'therapist': [],
        'client': []
    }
    for turn in dialogue:
        if turn['therapist_utterance']:
            formatted_dialogue.append(f"Therapist: {turn['therapist_utterance']}")
        if turn['client_utterance']:
            formatted_dialogue.append(f"Client: {turn['client_utterance']}")
        if turn['therapist_mi_code']:
            intent_label['therapist'].append(turn['therapist_mi_code'])
        if turn['client_mi_code']:
            intent_label['client'].append(turn['client_mi_code'])
    return formatted_dialogue, intent_label, result

def processed_csv_file(num):
    """
    Process reference CSV dialogue files
    Args:
        num (int): Session ID
    Returns:
        tuple: (Formatted dialogue list, intent label dictionary)
    """
    df = pd.read_csv(REFERENCE_DIALOGUE_PATH)
    dialogue = df[df['transcript_id'] == num]
    formatted_dialogue = []
    intent_label = {
        'therapist': [],
        'client': []
    }
    for index, row in dialogue.iterrows():
        if row['interlocutor'] == 'therapist':
            formatted_dialogue.append(f"Therapist: {row['utterance_text']}")
            intent_label['therapist'].append(row['main_therapist_behaviour'])
        else:
            formatted_dialogue.append(f"Client: {row['utterance_text']}")
            intent_label['client'].append(row['client_talk_type'])
    return formatted_dialogue, intent_label

if __name__ == "__main__":
    path = os.path.join(ROOT_DIR, "data/results/sessions/level1_by2llm/llama3.1:8b")
    try:
        format_dialogue, labels, dialogue = processed_json_file(1,path)
        print(f"Dialogue turns: {len(dialogue)}")
        print("First few dialogue turns:")
        for i, line in enumerate(dialogue[:5]):
            print(f"  {line}")
        print(labels)
    except Exception as e:
        print(f"Error processing CSV file: {str(e)}") 