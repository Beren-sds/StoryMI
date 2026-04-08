from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from .questionnaire_generation import QuestionnaireGenerator
import json
import os
import dotenv


def main(local_llm: bool = False):
    """main function"""
    # initialize the llm
    dotenv.load_dotenv()
    if local_llm:
        llm = ChatOllama(model="llama3.1:8b", temperature=0.7)
        print("Using local LLM: llama3.1:8b")
    else:
        llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=1.0,
            model_name="gpt-5-nano"
            # top_p=0.9 # not supported for gpt-5-nano
        )
    
    # initialize the generator
    generator = QuestionnaireGenerator(llm)
    
    for i in range(0, 1000):
        user_id = generator.generate_user(identity_type="Adult", i=i)
        print(f"Generated user{i+1}_id: {user_id}")
        result = generator.complete_questionnaire(user_id)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    

if __name__ == "__main__":
    main(local_llm=True)
