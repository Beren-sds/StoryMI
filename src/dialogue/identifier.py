from typing import List, Dict, Any
import json, re
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from ..questionnaire.config import DOMAIN_POOL
from ..utils.llm import initialize_llm
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()
class Symptoms(BaseModel):
    """
    Class for representing identified mental health symptoms
    """
    Symptom: str = Field(description="The identified mental health domain")
    Clue: str = Field(description="A direct quote from the dialogue that supports the identification of the symptom")

class Identifier:
    """
    Class for identifying mental health domains and symptoms from dialogue history
    """
    def __init__(self, model_name="llama3.1", temperature=0.7, local_llm=True):
        """
        Initialize Identifier
        """
        self.model_name = model_name
        self.temperature = temperature  
        self.llm = initialize_llm(local_llm=local_llm, model_name=model_name, temperature=temperature)
        self.parser = JsonOutputParser()
    
    def identify_domains(self, dialogue_history: List[Dict], domain_pool: List[str] = None) -> List[Dict]:
        """
        Identify mental health domains and symptoms from dialogue history
        Returns:
            List of dictionaries containing identified domains and clues
        """
        # 1) domain list
        domains = domain_pool if domain_pool is not None else DOMAIN_POOL
        domain_pool_str = ", ".join(domains) if isinstance(domains, list) else str(domains)
        
        # 2) system prompt
                # system_prompt = f"""You are a professional mental health counseling assistant. 
        #     Your task is to identify potential mental health domains from the conversation.

        #     Please carefully analyze the dialogue content, especially focusing on the client's statements, to identify:
        #     1. The involved mental health domains (from the specified list)
        #     2. Specific symptoms and corresponding quotes from the dialogue

        #     Use ONLY the following specified domains:
        #     {domain_pool_str}
        #     Do not invent new domains or modify domain names.

        #     Warning: ONLY return a valid JSON array as output, and NOTHING ELSE. 
        #     No explanations, no introductions, no notes. The output must be strictly like this:

        #     [
        #         {{
        #             "domain": "Depression",
        #             "clue": "I always feel sad and can't get interested in anything"
        #         }},
        #         {{
        #             "domain": "Anxiety",
        #             "clue": "I always worry about work issues and feel like I can't do things well"
        #         }}
        #     ]

        #     If no relevant domains are found, return an empty array: []
        #     """

        format_instructions = self.parser.get_format_instructions()

        # Use ONLY the following specified domains:
        # {domain_pool_str}
        # Do not invent new domains or modify domain names.

        system_prompt = f"""
        You are a professional mental-health assistant, 
        your task is to identify the symptoms of mental health issues from the conversation.
        For each identified symptoms, provide a **direct quote** (clue) from the dialogue that supports your identification.
        
        Warning: Output must be pure JSON array, nothing else.
        If none, output: []
        [
          {{
            "domain": "Anxiety",
            "clue": "I always worry about work issues..."
          }}
        ]
        """

        # 3) Split dialogue_history into chunks of 5 turns
        CHUNK_SIZE = 5
        chunks = [dialogue_history[i:i + CHUNK_SIZE]
                  for i in range(0, len(dialogue_history), CHUNK_SIZE)]
        recent_turns = dialogue_history[-5:] if len(dialogue_history) > 5 else dialogue_history       
       
        # dialogue_text = self._format_simple_dialogue(dialogue_history)
        # dialogue_text = self._format_simple_dialogue(recent_turns)

        chain = (self.llm | self.parser)
        merged: List[Dict[str, str]] = []

        for chunk in chunks:
            dialogue_text = self._format_simple_dialogue(chunk)
            # Prepare messages for the LLM
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Please analyze the following dialogue and identify potential symptoms:\n\n{dialogue_text}")
            ]
            
            try:
                # Use LLM and parser
                response = chain.invoke(messages)
                print("identifier response:", response)
                if isinstance(response, list):
                    merged.extend(response)
                elif isinstance(response, dict):
                    merged.append(response)
                else:
                    print(f"Unexpected response type: {type(response)} -> {response}")
            except Exception as e:
                print(f"Error identifying domains: {str(e)}")
            

        
        # 4) deduplicate by domain (first clue wins)
        dedup: Dict[str, str] = {}
        for item in merged:
            dom = item.get("domain")
            clue = item.get("clue")
            if dom and dom not in dedup:
                dedup[dom] = clue

        return [{"domain": d, "clue": c} for d, c in dedup.items()]
    
    def _format_simple_dialogue(self, dialogue_turns: List[Dict]) -> str:
        """
        Format dialogue turns into a simple text
        Args:
            dialogue_turns: List of dialogue turns    
        Returns:
            Formatted dialogue text
        """
        formatted_lines = []
        
        for turn in dialogue_turns:
            # Process standard format dialogue turns
            if "client_utterance" in turn and "therapist_utterance" in turn:
                if turn["client_utterance"]:
                    formatted_lines.append(f"Client: {turn['client_utterance']}")
                if turn["therapist_utterance"]:
                    formatted_lines.append(f"Therapist: {turn['therapist_utterance']}")
        
        if not formatted_lines:
            return "Dialogue history is empty"
            
        return "\n".join(formatted_lines)

if __name__ == "__main__":
    identifier = Identifier(model_name="qwen2.5:7b", temperature=0.1, local_llm=True)
    model_name = "llama3.1"
    sessions_dir = f"data/results/sessions/level1_by2llm/{model_name}"
    output = "data/results/sessions/level1_by2llm/reidentified_domains/" + model_name
    os.makedirs(output, exist_ok=True)
    counter = 0
    for i in range(1, 1000):
        f = f"session_{i}.json" 
        # f = f"questionnaire_user{i}.json"
        file_path = os.path.join(sessions_dir, f)
        # file_path = os.path.join(output, f)
    
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"File {f} does not exist. Skipping.")
            continue

        with open(file_path, "r") as file:
            session_data = json.load(file)
            # if session_data["identified_domains"] == []:
            #     print(f)
            #     print("Empty identified_domains")
            counter += 1
            identified_domains = identifier.identify_domains(session_data["dialogue_history"])
            session_data["identified_domains"] = identified_domains
            print(identified_domains)
            with open(os.path.join(output, f), "w") as outfile:
                json.dump(session_data, outfile, indent=2)

    print(f"Processed {counter} empty identified_domain files.")
