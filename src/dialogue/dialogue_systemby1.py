from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
import os
import dotenv
import time
from .identifier import Identifier

class DialogueSystemby1llm:
    def __init__(self, local_llm: bool = True, model_name: str = "llama3.2", max_turns: int = 25, temperature: float = 1, top_p: float = 0.9, identifier_model: str = "gpt-5-nano"):
        # Initialize LLM
        dotenv.load_dotenv()
        self.llm = self._initialize_llm(local_llm, model_name, temperature, top_p)
        # Initialize identifier 
        self.identifier = Identifier(model_name=identifier_model, temperature=0.3)
        self.user_info = None
        self.result1 = None


    def _initialize_llm(self, local_llm: bool, model_name: str, temperature: float = 0.7, top_p: float = 0.9):
        """Initialize the large language model"""
        if local_llm:
            return ChatOllama(model=model_name, temperature=temperature)
        else:
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=temperature,
                model_name=model_name
            )

    def process_user_data(self, user_data):
        """Process the user data."""
        user_info = user_data["user_info"].rsplit("_", 1)[0]
        questionnaire = user_data["questionnaire"]
        result1 = questionnaire["level1"]["result"]
        
        return user_info, result1

    def generate_dialogue(self, llm, user_data) -> str:
        self.user_info, self.result1 = self.process_user_data(user_data)
        parser = JsonOutputParser()
        system_prompt = f"""
        You need to simulate a dialogue between a therapist and a client. You will play both roles:
        
        1. Therapist: An experienced professional skilled in Motivational Interviewing (MI) techniques, using natural conversational language to communicate with the client.
        2. Client: A person with mental health issues, based on the following background information and assessment results.
        
        Client background information:
            {self.user_info}
        Client questionnaire assessment results:
            {self.result1}
        
        Alternate between therapist and client dialogue, starting with the therapist.(generate the therapist's opening statement to guide the client to discuss their most pressing concerns.)
        
        Therapist role requirements:
        1. Use MI techniques: emotional reflection, open-ended questions, affirmation, and amplifying discrepancies.
        2. Use different communication techniques each time, avoiding repetitive phrases or expressions.
        3. Use concise, conversational language, occasionally using brief responses or rhetorical questions.
        4. Focus on only one topic per inquiry.
        5. Avoid formulaic expressions like "It sounds like you feel..." or "How do you feel about..."
        
        Client role requirements:
        1. Display mental health issues and emotional states consistent with the background information and assessment results.
        2. Responses should be natural and authentic, sometimes hesitant, contradictory, or resistant.
        3. Gradually show motivation for change and insight, but the process should be realistic and gradual.
        4. Occasionally show resistance or unwillingness to share.
        
        Generate a complete counseling session with approximately 30 dialogue turns, showing the client's changes and the therapist's technique application.
        Please output in JSON format as follows:
    {{
        "dialogue_history": [
            {{
                "turn": 1
                "client_response": "Client dialogue",
                "therapist_response": "Therapist dialogue",
                "client_mi_code": "Client MI code",
                "therapist_mi_code": "Therapist MI code"
            }},
            {{
                "turn": 2,
                ....
            }}
            // ... other dialogue turns
        ]
    }}
        """
        
        # Create messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Please generate a complete dialogue simulation between a therapist and a client."}
        ]
        # print(messages)
        chain = llm | parser
        # Call LLM to generate dialogue
        response = chain.invoke(messages)
        
        # Return the generated dialogue
        return response
    

    def run_identifier(self, dialogue_history):
        """Run domain identification"""
        identified_domains = []
        try:
            identified_domains = self.identifier.identify_domains(dialogue_history)
            print(f"Identified domains: {identified_domains}")
        except Exception as e:
            print(f"Error identifying domains: {str(e)}")
        return identified_domains
    
    def run_session(self, user_data):
        """Run dialogue simulation and save results"""
        start_time = time.time()
        llm = self._initialize_llm(local_llm=False, model_name="gpt-4o", temperature=0.7)
        
        # Generate dialogue
        dialogue_result = self.generate_dialogue(llm=llm, user_data=user_data)
        dialogue_history = dialogue_result.get("dialogue_history", [])
        
        # Use run_identifier to identify domains in the dialogue
        identified_domains = self.run_identifier(dialogue_history)
        
        # Calculate session time
        end_time = time.time()
        
        # Add session metadata and identified domains
        result = {
            "domains": self.result1["domains"],
            "dialogue_history": dialogue_history,
            "session_metadata": {
                "duration_seconds": end_time - start_time,
                "session_naturally_ended": True,
                "end_reason": "completed"
            },
            "identified_domains": identified_domains,    
        }
        
        return result