"""
LLM tool module, provides initialization and configuration functions for language models.
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
import os
import dotenv
from dotenv import load_dotenv

load_dotenv()

# configuration
LLM_CONFIG = {
    "model_name": "llama3.1:8b",
    "temperature": 0.7,
}

def get_ollama_base_url():
    """Get Ollama base URL from environment variable, default to localhost"""
    ollama_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    # Ensure it starts with http://
    if not ollama_base_url.startswith("http://") and not ollama_base_url.startswith("https://"):
        ollama_base_url = f"http://{ollama_base_url}"
    return ollama_base_url

def initialize_llm(local_llm, model_name, temperature, top_p=None, n=None, reasoning_effort="minimal"):
    """
    Initialize LLM client
    
    Args:
        local_llm (bool): Whether to use a local LLM
        model_name (str): Model name, if not specified, use default configuration
        temperature (float): Temperature parameter controlling output randomness (not used for GPT-5 models)
        top_p (float, optional): Top-p sampling parameter (not used for GPT-5 models)
        n (int, optional): Number of completions
        reasoning_effort (str): For GPT-5 models only. Options: "minimal", "low", "medium", "high"
    
    Returns:
        LLM client object (ChatOpenAI or ChatOllama instance)
    """
    config = LLM_CONFIG.copy()
    
    if local_llm:
        # Local models (Ollama) support temperature and top_p
        return ChatOllama(
            base_url=get_ollama_base_url(),
            temperature=temperature if temperature is not None else config["temperature"],
            model=model_name or config["model_name"],
            top_p=top_p if top_p is not None else 1.0,
            n=n if n is not None else 1,
            timeout=300.0,  # 5 minutes timeout to prevent hanging
            request_timeout=300.0  # Request timeout
        )
    else:
        # OpenAI models
        model = model_name or config["model_name"]
        
        # Check if this is a GPT-5 model
        is_gpt5 = "gpt-5" in model.lower() or "o1" in model.lower() or "o3" in model.lower()
        
        if is_gpt5:
            # GPT-5/o1 models: do NOT pass temperature/top_p, use reasoning_effort instead
            print(f"   Initializing {model} with reasoning_effort={reasoning_effort}")
            print("   (temperature and top_p are not supported for reasoning models, use reasoning_effort instead)")
            
            return ChatOpenAI(
                model=model,
                api_key=os.getenv("OPENAI_API_KEY"),
                reasoning_effort=reasoning_effort  # Pass as explicit parameter, not in model_kwargs
            )
        else:
            # GPT-4, GPT-3.5, etc.: support temperature/top_p
            kwargs = {
                "model": model,
                "api_key": os.getenv("OPENAI_API_KEY"),
                "temperature": temperature if temperature is not None else config["temperature"],
            }
            
            # Only add top_p if it's provided
            if top_p is not None:
                kwargs["top_p"] = top_p
                
            return ChatOpenAI(**kwargs)
    
def initialize_embeddings(local_embeddings=False, model_name=None):
    """
    Initialize embeddings client
    
    Args:
        local_embeddings (bool): Whether to use a local embeddings model
        model_name (str): Model name, if not specified, use default configuration
    
    Returns:
        Embeddings client object (OpenAIEmbeddings or OllamaEmbeddings instance)
    """
    if local_embeddings:
        return OllamaEmbeddings(model=model_name)
    else:
        return OpenAIEmbeddings(model=model_name)
    