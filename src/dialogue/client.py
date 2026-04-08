"""
Client class with support for ablation experiments
Supports two modes: full_story, no_story
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
import os
import re
from typing import Dict, Any, List 


class Client:
    """Client agent for MI dialogue with ablation study support."""
    
    def __init__(self, llm, ablation_mode: str = "full_story"):
        """
        Initialize Client with ablation mode.
        
        Args:
            llm: Language model for generating responses
            ablation_mode: One of ["full_story", "no_story"]
                - full_story: Use complete background story (~200 words)
                - no_story: No background story (ablation condition)
        """
        self.llm = llm
        self.ablation_mode = ablation_mode
        
        # Validate ablation mode
        valid_modes = ["full_story", "no_story"]
        if ablation_mode not in valid_modes:
            raise ValueError(f"Invalid ablation_mode: {ablation_mode}. Must be one of {valid_modes}")
        
        self.user_info = None
        self.result1 = None
        self.result2 = None
        self.background_story = None
        self.system_prompt = None
        
        print(f"Client initialized with ablation_mode: {ablation_mode}")
    
    def process_user_data(self, user_data: Dict[str, Any]) -> None:
        """
        Process user data according to ablation mode.
        
        Args:
            user_data: Dictionary containing user information
                - For full_story: must contain "background_story"
                - For no_story: background_story will be set to empty string
        """
        if self.ablation_mode == "no_story":
            # Ablation: Remove all background story information
            self.background_story = ""
            print("   Using NO background story (ablation condition)")
        else:  # full_story (baseline)
            # Use complete background story
            self.background_story = user_data["background_story"]
            word_count = len(self.background_story.split())
            print(f"   Using FULL background story ({word_count} words)")

    def _build_message_history(self, dialogue_history, n: int = 6) -> str:
        """
        Build the message history for context.
        
        Args:
            dialogue_history: List of previous dialogue turns
            n: Number of recent turns to include
            
        Returns:
            Formatted history string
        """
        messages = []
        if not dialogue_history:
            return ""
            
        # Use up to the most recent n rounds of dialogue
        messages = dialogue_history[-n:] if len(dialogue_history) > n else dialogue_history
        history_text = "\n".join(messages) if messages else ""
        
        return history_text
    
    def _build_system_prompt(self) -> str:
        """
        Build system prompt based on ablation mode.
        
        Returns:
            System prompt string
        """
        # Base prompt template aligns with baseline prompt; no_story simply removes the story line
        if self.ablation_mode == "no_story":
            prompt = """You are a client receiving psychological counseling.

As a client, you task is to provide a conversation with the therapist according to your past traumatic experience and talk about your feelings and thoughts.
Constriants:
- Using the natrual and colloquial language, avoid metaphors and dramatic wording
- Only generate ONE utterance every time.
- Don't start with "It seems that" or "It sounds like" similar phrases.
"""
        else:  # full_story (baseline)
            prompt = f"""You are a client receiving psychological counseling.
This is your story/past traumatic experience: {self.background_story}

As a client, you task is to provide a conversation with the therapist according to your past traumatic experience and talk about your feelings and thoughts.
Constriants:
- Using the natrual and colloquial language, avoid metaphors and dramatic wording
- Only generate ONE utterance every time.
- Don't start with "It seems that" or "It sounds like" similar phrases.
"""

        return prompt
    
    def generate_response(
        self, 
        therapist_utterance: str, 
        dialogue_history: List[Dict] = None, 
        user_data: Dict[str, Any] = None
    ) -> str:
        """
        Generate client response to therapist's utterance.
        
        Args:
            therapist_utterance: The therapist's last utterance
            dialogue_history: Previous conversation turns
            user_data: User data (unused in current implementation)
            
        Returns:
            Client's response as a string
        """
        # Build system prompt according to ablation mode
        self.system_prompt = self._build_system_prompt()
        
        # Get formatted message history
        messages = self._build_message_history(dialogue_history)
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("system", f"Conversation history:\n{messages}"),
            ("human", "{therapist_utterance}")
        ])
            
        chain = prompt | self.llm
        response = chain.invoke({"therapist_utterance": therapist_utterance}).content
        
        # Print the response for debugging
        print(response)
        
        # Handle thinking tags for certain models (e.g., deepseek-r1)
        if getattr(self.llm, "model", "").lower().startswith("deepseek-r1"):
            response = re.sub(r"<think>.*?</think>\n?", "", response, flags=re.DOTALL)
        
        return response
    
    def reset(self) -> None:
        """Reset the client state."""
        self.user_info = None
        self.result1 = None
        self.result2 = None
        self.background_story = None
        self.system_prompt = None
        print(f"   Client reset (ablation_mode: {self.ablation_mode})")
