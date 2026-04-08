from .dialogue_system import DialogueSystem
from .dialogue_systemby1 import DialogueSystemby1llm
import json
import os
import time
from datetime import datetime
import traceback
import glob
from pathlib import Path
import argparse

SESSION_DIR = ""
# QUESTIONNAIRE_DIR = "data/results/questionnaires_processed"  # directory for questionnaire files
STORY_DIR = "data/results/background_stories"  # directory for background story files

def load_all_user_data(folder_path: str):
    """Load all user data from the specified directory"""
    questionnaires_dir = Path(folder_path)
    all_user_data = {}
    
    # check if the directory exists
    if not questionnaires_dir.exists():
        print(f"Warning: Questionnaire directory does not exist: {questionnaires_dir}")
        return all_user_data
    
    # get all questionnaire files
    questionnaire_files = list(questionnaires_dir.glob("questionnaire_user*.json"))
    story_files = list(questionnaires_dir.glob("background_story_*.json"))
    
    # if not questionnaire_files:
    #     print(f"Warning: No questionnaire files found in {questionnaires_dir}")
    #     return all_user_data
    if not story_files:
        print(f"Warning: No story files found in {questionnaires_dir}")
        return all_user_data
    
    # load questionnaire data for each user
    # for file_path in questionnaire_files:
    for file_path in story_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                user_data = json.load(f)
                # extract user ID from the file name
                # user_id = file_path.stem.split("_")[1]  # from "questionnaire_user1" extract "user1"
                user_id = file_path.stem.split("_")[2]  # from "background_story_user1" extract "user1"
                all_user_data[user_id] = user_data
        except Exception as e:
            print(f"Error loading questionnaire file {file_path}: {str(e)}")
    
    print(f"Successfully loaded questionnaire data for {len(all_user_data)} users")
    return all_user_data


def save_session_data(result, user_key):
    """Save session data to file"""
    session_file = f"{SESSION_DIR}/session_{user_key}.json"
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(session_file), exist_ok=True)
        
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump({
                "user_key": user_key,
                "total_turns": len(result["dialogue_history"]),
                **result,
            }, f, ensure_ascii=False, indent=2)
        print(f"Session data saved to: {session_file}")
        return session_file
    except Exception as save_error:
        print(f"Save failed: {str(save_error)}")
        return None


def process_users(start_index, end_index, system, all_user_data, user_keys):
    """Process users in the specified range"""
    global user_data  # declare as global variable, so it can be used in save_session_data
    
    success_count = 0
    total_to_process = end_index - start_index + 1

    
    for i in range(start_index, end_index + 1):
        user_key = user_keys[i]
        user_data = all_user_data[user_key]
        
        print(f"\n{'='*60}")
        print(f"Processing user {i-start_index+1}/{total_to_process} (index {i}: {user_key})")
        print(f"{'='*60}")
        
        try:
            print("\nStarting session...")
            result = system.run_session(user_data)
            
            save_session_data(result, user_key)
            success_count += 1
        
        except Exception as e:
            print(f"\nError processing user {user_key}: {str(e)}")
            print("\nDetailed error information:")
            traceback.print_exc()
    
    return success_count


def main(llm: int = None, args: argparse.Namespace = None):
    """Main function: load user data and run dialogue system
    
    Args:
        llm: LLM type (1 for single LLM, 2 for dual LLM) - for backward compatibility
        args: Parsed command line arguments (optional)
    """
    global SESSION_DIR
    
    # Parse arguments if provided, otherwise use defaults
    if args is None:
        parser = argparse.ArgumentParser(description='Run MI Dialogue Generation')
        parser.add_argument('--model_name', type=str, default='gpt-5-nano', help='Model name')
        parser.add_argument('--local_llm', action='store_true', help='Use local LLM')
        parser.add_argument('--max_turns', type=int, default=30, help='Maximum dialogue turns')
        parser.add_argument('--start_index', type=int, default=0, help='Start user index')
        parser.add_argument('--end_index', type=int, default=999, help='End user index (-1 for all)')
        parser.add_argument('--session_dir', type=str, default='data/results/sessions', help='Session directory')
        parser.add_argument('--ablation_mode', type=str, default='full_story', choices=['full_story', 'no_story'], 
                          help='Ablation mode: full_story (baseline) or no_story (ablation)')
        parser.add_argument('--use_mi_coding', action='store_true', default=True, 
                          help='Use MI coding (default: True, use --no_use_mi_coding to disable)')
        parser.add_argument('--no_use_mi_coding', dest='use_mi_coding', action='store_false',
                          help='Disable MI coding (ablation)')
        parser.add_argument('--llm_type', type=int, default=2, choices=[1, 2], 
                          help='LLM type: 1 for single LLM, 2 for dual LLM')
        args = parser.parse_args()
    
    # configure parameters
    config = {
        "start_index": args.start_index,
        "end_index": args.end_index,
        "use_local_llm": args.local_llm,
        "model_name": args.model_name,
        "max_turns": args.max_turns,
        "session_dir": args.session_dir,
        "ablation_mode": args.ablation_mode,
        "use_mi_coding": args.use_mi_coding
    }
    
    # Use llm parameter if provided (backward compatibility), otherwise use args.llm_type
    llm_type = llm if llm is not None else args.llm_type

    model_name = config["model_name"]
    story_dir = STORY_DIR  # Use the base directory directly
    
    if not os.path.exists(story_dir):
        print(f"   Story directory does not exist: {story_dir}")
        print(f"   Please run story generation first")
        return
    
    # Check if story files exist
    story_files = list(Path(story_dir).glob("background_story_*.json"))
    if not story_files:
        print(f"   No story files found in {story_dir}")
        print(f"   Please run story generation first")
        return
    
    print(f"   Using story directory: {story_dir}")
    print(f"   Found {len(story_files)} story files")
    
    ####################process user data####################
    print("================Loading user data=================")
    all_user_data = load_all_user_data(story_dir)
    # if no user data, exit
    if not all_user_data:
        print("Error: No user data found, please generate questionnaire data first")
        return
        
    user_keys = list(all_user_data.keys())
    user_keys.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else x)
    
    total_users = len(user_keys)
    
    print(f"Found {total_users} users")
    print(f"Processing users in order: {', '.join(user_keys)}")
    
    # determine the end index
    end_index = config["end_index"] if config["end_index"] >= 0 else total_users - 1
    start_index = config["start_index"]
    
    # validate the index range
    if start_index < 0 or start_index >= total_users:
        print(f"Error: start index {start_index} out of range (0-{total_users-1})")
        return
    if end_index >= total_users:
        print(f"Warning: end index {end_index} out of range, using max index {total_users-1}")
        end_index = total_users - 1
    print(f"Preparing to process user index range: {start_index} to {end_index} ({end_index-start_index+1}/{total_users} users)")


    ####################generate dialogue session####################
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if llm_type == 1:
        # create session directory
        # SESSION_DIR = config["session_dir"] + "/level1_by1llm" + f"/{config['model_name']}/{run_timestamp}" 
        SESSION_DIR = config["session_dir"] + "/level1_by1llm" + f"/{config['model_name']}"
        os.makedirs(SESSION_DIR, exist_ok=True)
        # initialize dialogue system
        print("================Initializing dialogue system=================")
        system = DialogueSystemby1llm(
            local_llm=config["use_local_llm"], 
            model_name=config["model_name"], 
            max_turns=config["max_turns"],
            # identifier_model=config["identifier_model"]  # use a stronger model for domain identification
        )
        success_count = process_users(start_index, end_index, system, all_user_data, user_keys)
    elif llm_type == 2:
        # Determine session directory based on ablation settings
        if config["ablation_mode"] == "no_story" and not config["use_mi_coding"]:
            session_subdir = "ablation_no_mi_coding_no_story"
        elif config["ablation_mode"] == "no_story":
            session_subdir = "ablation_no_story"
        elif not config["use_mi_coding"]:
            session_subdir = "ablation_no_mi_coding"
        else:
            session_subdir = "level1_by2llm"
        
        SESSION_DIR = os.path.join(config["session_dir"], session_subdir, config["model_name"])
        os.makedirs(SESSION_DIR, exist_ok=True)
        
        # initialize dialogue system
        print("================Initializing dialogue system=================")
        print(f"Ablation Mode: {config['ablation_mode']}")
        print(f"Use MI Coding: {config['use_mi_coding']}")
        
        # Determine temperature for GPT-5 models
        temperature = 1.0 if "gpt-5" in config["model_name"].lower() or "nano" in config["model_name"].lower() else 0.7
        
        system = DialogueSystem(
            local_llm=config["use_local_llm"], 
            model_name=config["model_name"], 
            max_turns=config["max_turns"],
            temperature=temperature,
            top_p=0.9,
            reasoning_effort="minimal",  # GPT-5 models
            ablation_mode=config["ablation_mode"],
            use_mi_coding=config["use_mi_coding"]
        )
        # run the dialogue system
        success_count = process_users(start_index, end_index, system, all_user_data, user_keys)
    else:
        print(f"Error: unsupported llm type: {llm_type}")
        return
    
    # print the summary
    print(f"\n{'='*60}")
    print(f"Completed: {success_count}/{end_index-start_index+1} users")
    print(f"{'='*60}")

def modify_session_dir():
    """
    Regenerate dialogue sessions only for users whose existing session file
    either does not exist or has "session_metadata.completed" != True.
    """
    global SESSION_DIR

    # --- configuration ---
    config = {
        "start_index": 0,
        "end_index": -1,
        "use_local_llm": True,
        "model_name": "phi4:14b",
        "max_turns": 30,
        "session_dir": "data/results/sessions"
    }

    # Story files are directly stored in background_stories folder (no subdirectories)
    story_dir = STORY_DIR  # Use the base directory directly
    
    if not os.path.exists(story_dir):
        print(f"   Story directory does not exist: {story_dir}")
        print(f"   Please run story generation first")
        return
    
    # Check if story files exist
    story_files = list(Path(story_dir).glob("background_story_*.json"))
    if not story_files:
        print(f"   No story files found in {story_dir}")
        print(f"   Please run story generation first")
        return
    
    print(f"   Using story directory: {story_dir}")
    print(f"   Found {len(story_files)} story files")

    # --- load background stories ---
    print("================ Loading user data ================")
    all_user_data = load_all_user_data(story_dir)
    if not all_user_data:
        print("Error: No user data found, please generate questionnaire data first")
        return

    user_keys = sorted(
        all_user_data.keys(),
        key=lambda x: int("".join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else x,
    )
    total_users = len(user_keys)
    print(f"Found {total_users} users: {', '.join(user_keys)}")

    # --- validate start / end index ---
    start_index = config["start_index"]
    end_index = config["end_index"] if config["end_index"] >= 0 else total_users - 1
    if start_index < 0 or start_index >= total_users:
        print(f"Error: start index {start_index} out of range (0-{total_users-1})")
        return
    if end_index >= total_users:
        print(f"Warning: end index {end_index} out of range, using max index {total_users-1}")
        end_index = total_users - 1
    print(f"Processing users {start_index}‒{end_index} "
          f"({end_index - start_index + 1}/{total_users})")

    # --- set up SESSION_DIR ---
    SESSION_DIR = os.path.join(
        config["session_dir"], "level1_by2llm" , config["model_name"]
    )
    os.makedirs(SESSION_DIR, exist_ok=True)
    print(f"SESSION_DIR set to: {SESSION_DIR}")

    # --- initialise LLM system ---
    print("================ Initialising dialogue system ================")
    system = DialogueSystem(
        local_llm=config["use_local_llm"],
        model_name=config["model_name"],
        max_turns=config["max_turns"],
        temperature=0.7,
        top_p=0.9,
        reasoning_effort="minimal"  # GPT-5 models
    )

    # --- process (or skip) users ---
    success_count = 0
    for idx in range(start_index, end_index + 1):
        user_key = user_keys[idx]
        session_path = os.path.join(SESSION_DIR, f"session_{user_key}.json")

        # check existing session file
        regenerate = True
        if os.path.exists(session_path):
            try:
                with open(session_path, "r", encoding="utf-8") as f:
                    session_json = json.load(f)
                if session_json.get("session_metadata", {}).get("completed") is True:
                    regenerate = False
                    print(f"[DONE] User {user_key}: session already completed — skipping.")
                elif len(session_json.get("dialogue_history", [])) >= system.max_turns:
                    regenerate = False
                    print(f"[DONE] User {user_key}: session has enough turns — skipping.")
                else:
                    print(f"[RETRY] User {user_key}: session incomplete — regenerating.")

                    #test
                    # success_count += 1
            except Exception as e:
                print(f"   User {user_key}: could not read session file ({e}) — regenerating.")
        else:
            print(f"   User {user_key}: no existing session — creating new session.")

        # run or skip
        if regenerate:
            try:
                result = system.run_session(all_user_data[user_key])
                save_session_data(result, user_key)
                success_count += 1
            except Exception as e:
                print(f"Error processing user {user_key}: {e}")
                traceback.print_exc()

    print(f"\n======= Finished: regenerated/created {success_count} sessions =======")


if __name__ == "__main__":
    # Support both old style (llm parameter) and new style (command line args)
    import sys
    if len(sys.argv) > 1:
        # Use argparse for command line arguments
        parser = argparse.ArgumentParser(description='Run MI Dialogue Generation')
        parser.add_argument('--model_name', type=str, default='gpt-5-nano', help='Model name')
        parser.add_argument('--local_llm', action='store_true', help='Use local LLM')
        parser.add_argument('--max_turns', type=int, default=30, help='Maximum dialogue turns')
        parser.add_argument('--start_index', type=int, default=0, help='Start user index')
        parser.add_argument('--end_index', type=int, default=999, help='End user index (-1 for all)')
        parser.add_argument('--session_dir', type=str, default='data/results/sessions', help='Session directory')
        parser.add_argument('--ablation_mode', type=str, default='full_story', choices=['full_story', 'no_story'], 
                          help='Ablation mode: full_story (baseline) or no_story (ablation)')
        parser.add_argument('--use_mi_coding', action='store_true', default=True, 
                          help='Use MI coding (default: True, use --no_use_mi_coding to disable)')
        parser.add_argument('--no_use_mi_coding', dest='use_mi_coding', action='store_false',
                          help='Disable MI coding (ablation)')
        parser.add_argument('--llm_type', type=int, default=2, choices=[1, 2], 
                          help='LLM type: 1 for single LLM, 2 for dual LLM')
        args = parser.parse_args()
        main(args=args)
    else:
        # Backward compatibility: use default parameters
        main(llm=2)