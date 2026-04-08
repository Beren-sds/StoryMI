from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List
import json, re

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Import shared components
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.llm import initialize_llm
from src.utils.retriever import DocumentRetriever

MI_FILE_PATH: Path = Path("data/MI/misc3.pdf")
MI_CODE_PATH: Path = Path("src/dialogue/mi_code.json")

class TherapistResponse(BaseModel):
    therapist_utterance: str = Field(description="The therapist's utterance content")
    client_mi_code: str = Field(description="MI code corresponding to the client's utterance")
    therapist_mi_code: str = Field(description="MI code should be used to reply client's utterance(reflection, question, or therapist_input)")

class MIPrediction(BaseModel):
    client_mi_code: str = Field(description="MI code corresponding to the client's utterance")
    therapist_mi_code: str = Field(description="MI code should be used to reply client's utterance")


class Therapist:
    def __init__(self, llm, llm_json_mode):
        self.llm = llm if llm is not None else initialize_llm()
        self.llm_json_mode = llm_json_mode
        self.use_mi_coding = use_mi_coding  # Store MI coding flag
            
        self.system_prompt = None
        self.user_info = None
        self.result1 = None
        self.result2 = None
        self.domains = {}
        self.questionnaire = None
        self.initialized = False
        self.debug = False

        # DocumentRetriever
        # self.document_retriever = DocumentRetriever(MI_FILE_PATH)
        
        # Initialize output parser
        # self.parser = JsonOutputParser(pydantic_object=TherapistResponse)
        self.parser = JsonOutputParser(pydantic_object=MIPrediction)
        
        # Load MI codes from JSON file
        self.mi_codes = self._load_mi_codes()
    
    def _load_mi_codes(self):
        """Load MI codes from the JSON file"""
        try:
            with open(MI_CODE_PATH, 'r') as file:
                return json.load(file)
        except Exception as e:
            print(f"Error loading MI codes: {e}")
            return {}
    
    
    def _build_message_history(self, dialogue_history, n: int = 5) -> List:
        """Build the message history."""
        messages = []
        if not dialogue_history:
            return messages
            
        # Use up to the most recent n rounds of dialogue
        messages = dialogue_history[-n:] if len(dialogue_history) > n else dialogue_history
        # history_text = "\n".join(messages) if messages else ""
        
        # return history_text
        return messages
    
    def mi_code_detection(self, messages, client_utterance):
        """
        Detect MI codes for client utterance and predict therapist MI technique.
        
        This method is used in 'with_mi_coding' mode for generation guidance,
        and also for post-hoc annotation in 'without_mi_coding' mode.
        """
        format_instructions = self.parser.get_format_instructions()

        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert MI code annotator. 
            Based on the MI code definitions and the conversation history, your task is to:
                - Identify the most appropriate MI code for the client's last utterance.
                - Select the MI technique the therapist should use next.
       
            MI Code Definitions:
            {mi_codes}
            Conversation History:
            {messages}
            
            Constraints:
            - Crucially, vary your MI techniques throughout the conversation. Avoid relying heavily on only one or two types of responses (like simple reflections). 
            - Consider the current phase of the conversation when selecting techniques:
                - Early: More open questions and reflections to build rapport and understanding
                - Middle: More complex reflections and strategic questions to explore ambivalence
                - Later: information giving, and advice to support the client in making decisions
            
            Output Format: Return the result strictly in valid JSON format, without any additional text or explanation.:
            {{
                "client_mi_code": "client_mi_code",
                "therapist_mi_code": "therapist_mi_code"
            }}
            """),
            ("human", "{client_utterance}")
        ])

        chain = (
            prompt          
            | self.llm_json_mode
            # | self.parser
        )

        response = chain.invoke({
            "messages": messages,
            "mi_codes": self.mi_codes,
            "client_utterance": client_utterance,
            # "format_instructions": format_instructions
        })
        response = json.loads(response.content)
        print(response) 
        return response
    
    def generate_utterance(self, messages, therapist_micode, wrap_up_instruction, client_utterance):
        # parser = JsonOutputParser()
        # format_instructions = parser.get_format_instructions()
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an experienced psychotherapist skilled in Motivational Interviewing (MI) techniques.  
            {wrap_up_instruction}
            Given the conversational context and the mi_code, your task is to generate your response that strictly follows the selected MI code.
            
             The Constriants of your response:
                - Please only generate 1-2 utterance every time, using casual, simple natural language.
                - Don't always use the repetitive sentence patterns.
                - Avoid starting the sentences with "It seems that", "It sounds like" or similar phrases.
                - When using the "reflection" technique, express your ability of empathy and understanding, and avoid using the same sentence patterns as client.
            
            Here is the difinition of MI code you can refer to:
                {mi_codes}
            Here is the selected MI code to generate your response:
                {therapist_micode}
            Conversation history:
                {messages}       
            Only return the content of therapist response, without any additional text or explanation.
            """),
            ("human", "{client_utterance}")
        ])

        
        chain = (
            prompt          
            | self.llm 
        )
        
        response = chain.invoke({
            "messages": messages,
            "therapist_micode": therapist_micode,
            # "mi_context": mi_context,
            "mi_codes": self.mi_codes,
            "client_utterance": client_utterance,
            "therapist_micode": therapist_micode,
            # "format_instructions": format_instructions,
            "wrap_up_instruction": wrap_up_instruction
        }).content
        print(response)
        # if getattr(self.llm, "model", "").lower().startswith("deepseek-r1"):
            # response = re.sub(r"<think>.*?</think>\n?", "", response, flags=re.DOTALL)
        
        return response
    
    def generate_response(self, client_utterance: str, dialogue_history: List[Dict] = None, is_wrapping_up: bool = False) -> Dict:

        messages = self._build_message_history(dialogue_history)
        wrap_up_instruction = ""
        if is_wrapping_up:
            # mi_code = self.mi_code_detection(messages, client_utterance)
            wrap_up_instruction = """
            IIMPORTANT: The conversation is nearing its end. Your response MUST move the conversation toward completion rather than opening new topics.
            Based on the dialogue history, You MUST end the session naturally by:
            1. Summarizing key insights or progress made during the conversation
            2. Moving toward a natural conclusion of the session, and don't leave any open-ended questions.
            3. Say goodbye to the client
            Please only generate 1-2 utterance, using casual, simple natural language
            """
            print("Wrap-up instruction added.")
            # prompt = wrap_up_instruction.format(messages=messages)
            # therapist_utterance = self.llm.invoke(prompt).content
        # else:
        mi_code = self.mi_code_detection(messages, client_utterance)
        therapist_utterance = self.generate_utterance(messages, mi_code, wrap_up_instruction, client_utterance)
        # if isinstance(therapist_utterance, dict) and "properties" in therapist_utterance:
            # therapist_utterance = {k: therapist_utterance["properties"][k]["description"] for k in therapist_utterance["properties"]}
        # print(mi_code)
        # print(therapist_utterance)
        
        return {
            "client_mi_code": mi_code["client_mi_code"],
            "therapist_mi_code": mi_code["therapist_mi_code"],
            "therapist_utterance": therapist_utterance
        }
        

if __name__ == "__main__":
    # llm = ChatOpenAI(
    #             api_key=os.getenv("OPENAI_API_KEY"),
    #             temperature=0.7,
    #             model_name="gpt-4o-mini"
    #         )
    llm = ChatOllama(
            temperature=0.7,
            model="llama3.1",
            # top_p=top_p if top_p is not None else config.get("top_p", 1.0),
        )
    therapist = Therapist(llm=llm)
    result = therapist.generate_response("How are you feeling today?", dialogue_history=None, is_wrapping_up=False)
    print(result)

