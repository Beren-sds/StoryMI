import os,sys
import json
from pprint import pprint
import time
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from src.utils.llm import initialize_llm
from src.questionnaire.config import QUESTIONNAIRE_DIR

class UserProfile(BaseModel):
    basic_info: str = Field(description="Basic profile: age, occupation/education, family situation, social status")
    life_history: str = Field(description="Life history: upbringing/family background and significant milestones")
    recent_events: str = Field(description="Events in the last 6–12 months that may have affected mental health")
    personality: str = Field(description="Core personality traits and typical coping strategies")

class UserProfileGenerator:
    """Class for generating user profiles based on questionnaire data."""
    def __init__(self, model_name="llama3.1:8b", temperature=0.7, local_llm=True):

        self.llm = initialize_llm(model_name=model_name, temperature=temperature, local_llm=local_llm)
        self.system_prompt = """
        Based on the following questionnaire screening results, write a first-person narrative telling a story (less than 200 words).
        
        The Requirements of the story:
        - Choose one primary symptom (the one with the most severe level in the questionnaire results).
        - Focus on ONE specific real-life scene (e.g., at work, during dinner, morning routine).
        - Describe concrete actions and behaviors: what the person did, what happened around them, what objects they interacted with.
        - Show how the symptom directly disrupts or alters a normal activity using plain language. Use short, direct sentences with minimal adjectives.
        - Maintain a straightforward, matter-of-fact tone as if speaking during a therapy session.
        - Describe emotions factually ("I felt anxious" rather than "crushing waves of anxiety overwhelmed me").


        
        The questionnaire screening results is as follows: {results}
        The user's explanations of the answers are as follows: {user_response}
        Only return the story without any additional text or explanation.
        """
        
    def load_questionnaire_data(self, user_id: str, questionnaire_path: str = None):
        """Load questionnaire data from a JSON file."""
        if questionnaire_path is None:
            questionnaire_path = os.path.join(QUESTIONNAIRE_DIR, f"questionnaire_user{user_id}.json")
        
        with open(questionnaire_path, 'r') as file:
            questionnaire_data = json.load(file)
        
        return questionnaire_data["questionnaire"]
    
    def generate(self, questionnaire_data):
        """Generate user profile from questionnaire data."""
        
        user_response = questionnaire_data["level1"]["user_response"]["explanations"]
        screening_results = questionnaire_data["level1"]["result"]

        parser = JsonOutputParser(pydantic_object=UserProfile)
        format_instructions = parser.get_format_instructions()
        # pprint(format_instructions)

        prompt = PromptTemplate(
            template=self.system_prompt,
            input_variables=["results", "user_response"],
            # partial_variables={"format_instructions": format_instructions}
        )
    
        chain = prompt | self.llm

        inputs = {
            "results": screening_results,
            "user_response": user_response 
        }
        
        response = chain.invoke(inputs)
        return {
            "background_story": response.content,
            "user_response": user_response,
            "screening_results": screening_results
        }

    
    def save_to_path(self, user_id, profile_data, output_dir=None):

        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(QUESTIONNAIRE_DIR), "background_stories")
        
        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create the output filename
        output_file = os.path.join(output_dir, f"background_story_{user_id}.json")
        
        # Convert to dict if it's a Pydantic model
        if hasattr(profile_data, "model_dump"):
            profile_data = profile_data.model_dump()
        
        # Write the data to a JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(profile_data, f, ensure_ascii=False, indent=4)
        
        print(f"User profile saved to {output_file}")
        return output_file


        
if __name__ == "__main__":


    # generator = UserProfileGenerator(model_name="gpt-5-nano", temperature=0.7, local_llm=False)
    model_name = "llama3.1:8b"
    generator = UserProfileGenerator(model_name=model_name, temperature=0.7, local_llm=True)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(QUESTIONNAIRE_DIR), f"background_stories/{model_name}_batch_{timestamp}")

    for i in range(1, 1001):
        start_time = time.time()
        print(f"Processing user {i}...")
        print(f"\n{'='*60}")
        print(f"Generating story {i}/1000...")
        print(f"{'='*60}")
        user_id = str(i)
        questionnaire_path = os.path.join(QUESTIONNAIRE_DIR, f"questionnaire_user{user_id}.json")
        
        if not os.path.exists(questionnaire_path):
            print(f"File {questionnaire_path} does not exist. Skipping.")
            continue
        
        questionnaire_data = generator.load_questionnaire_data(user_id, questionnaire_path)
        # pprint(questionnaire_data)
        
        if questionnaire_data:
            result = generator.generate(questionnaire_data)
            print("Generated User Profile:")    
            pprint(result["background_story"])

            generator.save_to_path(user_id, result, output_dir=output_dir)
            print(f"Story {i} generated")
            end_time = time.time()
            print(f"Time taken: {end_time - start_time:.2f} seconds")
        else:
            print("No questionnaire data found.")

    print(f"\n{'='*60}")
    print(f"All stories saved to: {output_dir}/")
    print(f"{'='*60}")

