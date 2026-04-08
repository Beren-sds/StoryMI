"""
Unified script for running ablation experiments
Supports both story ablation and MI coding ablation experiments
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import traceback
import time

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from src.dialogue.dialogue_system import DialogueSystem


def parse_args():
    """Parse command line arguments for ablation experiments."""
    parser = argparse.ArgumentParser(
        description='Run MI Dialogue Generation with Ablation Study Support'
    )
    
    # Model configuration
    parser.add_argument(
        '--model_name', 
        type=str, 
        default='gpt-5-nano',
        help='Name of the model to use (default: gpt-5-nano)'
    )
    parser.add_argument(
        '--local_llm',
        action='store_true',
        help='Use local LLM instead of API'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Temperature for generation (default: 1.0 for GPT-5 models)'
    )
    parser.add_argument(
        '--max_turns',
        type=int,
        default=30,
        help='Maximum number of dialogue turns (default: 30)'
    )
    
    # Ablation configuration
    parser.add_argument(
        '--ablation_mode',
        type=str,
        default='full_story',
        choices=['full_story', 'no_story'],
        help='Story ablation mode: full_story (baseline) or no_story (ablation)'
    )
    parser.add_argument(
        '--use_mi_coding',
        action='store_true',
        default=True,
        help='Use MI coding (default: True, use --no_use_mi_coding to disable)'
    )
    parser.add_argument(
        '--no_use_mi_coding',
        dest='use_mi_coding',
        action='store_false',
        help='Disable MI coding (ablation)'
    )
    
    # Data paths
    parser.add_argument(
        '--story_dir',
        type=str,
        default='data/results/background_stories',
        help='Directory containing background stories'
    )
    parser.add_argument(
        '--session_dir',
        type=str,
        default=None,
        help='Output directory for dialogue sessions (auto-generated if not specified)'
    )
    
    # Processing range
    parser.add_argument(
        '--start_index',
        type=int,
        default=0,
        help='Start index for processing users (default: 0)'
    )
    parser.add_argument(
        '--end_index',
        type=int,
        default=-1,
        help='End index for processing users, -1 for all (default: -1)'
    )
    
    return parser.parse_args()


def load_user_data(user_id: str, story_dir: str, ablation_mode: str):
    """
    Load user data according to ablation mode.
    
    Args:
        user_id: User identifier
        story_dir: Directory containing full stories
        ablation_mode: Mode of ablation experiment
        
    Returns:
        Dictionary containing user data
    """
    story_file = Path(story_dir) / f"background_story_{user_id}.json"
    if not story_file.exists():
        raise FileNotFoundError(f"Story file not found: {story_file}")
    
    with open(story_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # Always load all data from the story file
        user_data = {
            "background_story": data.get("background_story", ""),
            "screening_results": data.get("screening_results", {}),
            "user_response": data.get("user_response", [])
        }
    
    return user_data


def save_session_data(result: dict, user_id: str, session_dir: str):
    """
    Save dialogue session data to file.
    
    Args:
        result: Session result dictionary
        user_id: User identifier
        session_dir: Output directory
    """
    os.makedirs(session_dir, exist_ok=True)
    session_file = os.path.join(session_dir, f"session_{user_id}.json")
    
    try:
        # Add user_key and total_turns before dialogue_history
        dialogue_history = result.get("dialogue_history", [])
        total_turns = len(dialogue_history)
        
        # Create the output structure with correct field order:
        # user_key, total_turns, dialogue_history, domains, identified_domains, session_metadata
        output_data = {
            "user_key": str(user_id),
            "total_turns": total_turns,
            "dialogue_history": dialogue_history,
            "domains": result.get("domains", {}),
            "identified_domains": result.get("identified_domains", []),
            "session_metadata": result.get("session_metadata", {})
        }
        
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"   Session saved: {session_file}")
    except Exception as e:
        print(f"   Error saving session: {str(e)}")


def determine_session_dir(args):
    """
    Determine session output directory based on ablation settings.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Session directory path
    """
    if args.session_dir is None:
        # Auto-generate based on ablation mode
        if args.ablation_mode == "no_story" and not args.use_mi_coding:
            session_dir = f"data/results/sessions/ablation_no_mi_coding_no_story/{args.model_name}"
        elif args.ablation_mode == "no_story":
            session_dir = f"data/results/sessions/ablation_no_story/{args.model_name}"
        elif not args.use_mi_coding:
            session_dir = f"data/results/sessions/ablation_no_mi_coding/{args.model_name}"
        else:  # full_story and use_mi_coding (baseline)
            session_dir = f"data/results/sessions/level1_by2llm/{args.model_name}"
    else:
        session_dir = args.session_dir
    
    return session_dir


def main():
    """Main function for running ablation experiments."""
    
    # Parse arguments
    args = parse_args()
    
    # Print configuration
    print("=" * 80)
    print("MI Dialogue Generation - Ablation Study")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Story Ablation Mode: {args.ablation_mode}")
    print(f"MI Coding: {'Enabled' if args.use_mi_coding else 'Disabled (Ablation)'}")
    print(f"Max Turns: {args.max_turns}")
    print(f"Temperature: {args.temperature}")
    print(f"Story Directory: {args.story_dir}")
    print("=" * 80)
    
    # Determine session output directory
    session_dir = determine_session_dir(args)
    
    print(f"Output Directory: {session_dir}")
    print("=" * 80)
    
    # Create output directory if it doesn't exist
    os.makedirs(session_dir, exist_ok=True)
    print(f"Output directory created/verified: {session_dir}")
    print("=" * 80)
    
    # Get list of user IDs
    story_dir = Path(args.story_dir)
    story_files = sorted(story_dir.glob("background_story_*.json"))
    
    if not story_files:
        print(f"No story files found in {args.story_dir}")
        return
    
    # Extract user IDs
    user_ids = []
    for f in story_files:
        # Extract user ID from filename (e.g., background_story_1.json -> "1")
        user_id = f.stem.split("_")[-1].replace("user", "")
        user_ids.append(user_id)
    
    user_ids.sort(key=lambda x: int(x) if x.isdigit() else x)
    total_users = len(user_ids)
    
    print(f"Found {total_users} users")
    
    # Determine processing range
    start_idx = args.start_index
    end_idx = args.end_index if args.end_index >= 0 else total_users - 1
    
    if start_idx < 0 or start_idx >= total_users:
        print(f"Invalid start_index: {start_idx} (must be 0-{total_users-1})")
        return
    
    if end_idx >= total_users:
        print(f"[WARNING] end_index {end_idx} exceeds total users, using {total_users-1}")
        end_idx = total_users - 1
    
    users_to_process = user_ids[start_idx:end_idx+1]
    print(f"Processing users {start_idx} to {end_idx} ({len(users_to_process)} users)")
    print("=" * 80)
    
    # Initialize dialogue system with ablation parameters
    print("\nInitializing Dialogue System...")
    dialogue_system = DialogueSystem(
        local_llm=args.local_llm,
        model_name=args.model_name,
        max_turns=args.max_turns,
        temperature=args.temperature,
        ablation_mode=args.ablation_mode,  # Pass ablation mode
        use_mi_coding=args.use_mi_coding  # Pass MI coding flag
    )
    print("Dialogue system initialized")
    
    # Process each user
    successful = 0
    failed = 0
    skipped = 0
    
    for idx, user_id in enumerate(users_to_process, 1):
        global_idx = start_idx + idx
        print(f"\n{'='*80}")
        print(f"Processing User {user_id} ({idx}/{len(users_to_process)})")
        print(f"{'='*80}")
        
        # Check if session already exists and is complete
        session_path = os.path.join(session_dir, f"session_{user_id}.json")
        should_skip = False
        
        if os.path.exists(session_path):
            try:
                with open(session_path, 'r', encoding='utf-8') as f:
                    session_json = json.load(f)
                
                # Check if session is completed
                if session_json.get("session_metadata", {}).get("completed") is True:
                    should_skip = True
                    print(f"[SKIP] User {user_id}: session already completed — skipping.")
                # Check if session has enough turns (fallback check)
                elif len(session_json.get("dialogue_history", [])) >= dialogue_system.max_turns:
                    should_skip = True
                    print(f"[SKIP] User {user_id}: session has enough turns ({len(session_json.get('dialogue_history', []))} turns) — skipping.")
                else:
                    print(f"[REGEN] User {user_id}: session incomplete — regenerating.")
            except Exception as e:
                print(f"[WARNING] User {user_id}: could not read session file ({e}) — regenerating.")
        else:
            print(f"[NEW] User {user_id}: no existing session — creating new session.")
        
        # Skip if session already exists and is complete
        if should_skip:
            skipped += 1
            continue
        
        try:
            start_time = time.time()
            
            # Load user data according to ablation mode
            print(f"Loading user data (mode: {args.ablation_mode})...")
            user_data = load_user_data(
                user_id=user_id,
                story_dir=args.story_dir,
                ablation_mode=args.ablation_mode
            )
            print(f"User data loaded")
            
            # Generate dialogue
            print(f"Generating dialogue session...")
            result = dialogue_system.run_session(user_data)
            
            # Save session
            save_session_data(result, user_id, session_dir)
            
            elapsed = time.time() - start_time
            successful += 1
            
            print(f"\nSession completed for user {user_id}")
            turn_count = len(result.get('dialogue_history', []))
            session_metadata = result.get('session_metadata', {})
            print(f"   Turns: {turn_count}")
            print(f"   Naturally ended: {session_metadata.get('session_naturally_ended', False)}")
            print(f"   Time: {elapsed:.2f}s")
            
        except Exception as e:
            failed += 1
            print(f"\nError processing user {user_id}")
            print(f"   Error: {str(e)}")
            traceback.print_exc()
            continue
    
    # Final summary
    print("\n" + "=" * 80)
    print("Ablation Experiment Complete")
    print("=" * 80)
    print(f"Story Ablation Mode: {args.ablation_mode}")
    print(f"MI Coding: {'Enabled' if args.use_mi_coding else 'Disabled (Ablation)'}")
    print(f"Successful: {successful}/{len(users_to_process)}")
    print(f"[SKIP] Skipped: {skipped}/{len(users_to_process)}")
    print(f"Failed: {failed}/{len(users_to_process)}")
    print(f"Sessions saved to: {session_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
