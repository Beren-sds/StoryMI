import traceback
from typing import Dict, List, Annotated, TypedDict, Literal, Any
from langgraph.graph import StateGraph, END
import operator
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from .client import Client
from .therapist import Therapist
from .identifier import Identifier
from langchain_core.output_parsers import JsonOutputParser
import os
import dotenv
import time
from ..utils.llm import initialize_llm
from pprint import pprint
import re

class AgentState(TypedDict):
    user_data: Dict
    messages: List
    dialogue_history: List[Dict]
    current_speaker: Literal["client", "therapist"]
    is_complete: bool
    client_utterance: str
    therapist_utterance: str
    turn_count: int
    identified_domains: List[Dict]
    is_wrapping_up: bool  # Added to signal that conversation should wrap up

class DialogueSystem:
    def __init__(self, 
                 local_llm: bool = True, 
                 model_name: str = "llama3.1", 
                 max_turns: int = 30, 
                 temperature: float = 0.6, 
                 top_p: float = 0.9,
                 reasoning_effort: str = "minimal",  # for GPT-5 models
                 ablation_mode: str = "full_story",  # "full_story" or "no_story"
                 use_mi_coding: bool = True  # Enable/disable MI coding
                 ):
        # Initialize LLM
        dotenv.load_dotenv()
        
        # Check if using GPT-5 model
        self.is_gpt5 = "gpt-5" in model_name.lower() or "nano" in model_name.lower() or "o3" in model_name.lower()
        self.model_name = model_name
        
        # Initialize main LLMs
        self.therapist_llm = initialize_llm(
            local_llm, model_name, temperature, top_p, 
            reasoning_effort=reasoning_effort
        )
        self.client_llm = initialize_llm(
            local_llm, model_name, temperature, top_p,
            reasoning_effort=reasoning_effort
        )
        
        # JSON mode: Only use Ollama for local models
        if local_llm:
            from ..utils.llm import get_ollama_base_url
            self.llm_json_mode = ChatOllama(
                base_url=get_ollama_base_url(),
                model=model_name, 
                temperature=0.3, 
                format="json"
            )
        else:
            # For OpenAI API, use json_mode in the API call instead
            if self.is_gpt5:
                self.llm_json_mode = initialize_llm(
                    local_llm, model_name, 1.0, None,
                    reasoning_effort="minimal"  # Lower effort for JSON tasks
                )
            else:
                self.llm_json_mode = ChatOpenAI(
                    model=model_name, 
                    temperature=1.0,
                    model_kwargs={"response_format": {"type": "json_object"}}
                )

        # Initialize client and therapist with ablation parameters
        self.client = Client(self.client_llm, ablation_mode=ablation_mode)
        self.therapist = Therapist(self.therapist_llm, self.llm_json_mode, use_mi_coding=use_mi_coding)

        # Completion detector
        self.completion_detector_llm = initialize_llm(
            local_llm=local_llm,
            model_name=model_name,
            temperature=0.7,
            reasoning_effort="minimal" if self.is_gpt5 else None
        )
        
        # Initialize identifier 
        self.identifier = Identifier(
            model_name=model_name, 
            temperature=0.3, 
            local_llm=local_llm,
            reasoning_effort=reasoning_effort if self.is_gpt5 else None
        )
        
        # Store max_turns
        self.max_turns = max_turns
        self.session_metadata = {"session_naturally_ended": False, "end_reason": ""}
        
        # Initialize workflow
        self.workflow = self._create_workflow()
        self.current_state = None

    ####################### Workflow Definition #########################
    def _create_workflow(self) -> StateGraph:
        """Create the dialogue workflow"""
        
        # Create the state graph
        workflow = StateGraph(AgentState)
        
        workflow.add_node("therapist_agent", self._run_therapist_agent)
        workflow.add_node("client_agent", self._run_client_agent)
        workflow.add_node("completion_detector", self._check_completion)
        workflow.add_node("identifier", self._run_identifier)
        workflow.add_node("end_session", self._end_session)
        
        workflow.set_entry_point("therapist_agent")
        
        workflow.add_edge("therapist_agent", "completion_detector")
        workflow.add_conditional_edges(
        "completion_detector",
        self._should_complete,
        {
            False: "client_agent",
            True: "identifier"
        }
        )
        # workflow.add_edge("completion_detector", "client_agent")
        workflow.add_edge("client_agent", "therapist_agent")
       

        workflow.add_edge("identifier", "end_session")
        workflow.add_edge("end_session", END)
        
        graph = workflow.compile()
        image_data = graph.get_graph().draw_mermaid_png()
        with open("workflow.png", "wb") as f:
            f.write(image_data)
        return graph

    

    ####################### Define Workflow Nodes #########################
    
    def _run_therapist_agent(self, state: AgentState) -> AgentState:
        """Run therapist agent"""
        # Get current client utterance
        client_utterance = state["client_utterance"]
        
        # First turn check
        if state["turn_count"] == 0 and not state["messages"]:
            therapist_result = {
                "therapist_utterance": "Hi there! I'm here to help you. How are you feeling today?",
                "client_mi_code": "",
                "therapist_mi_code": "question"  
            }

        else:
        # Others
            therapist_result = self.therapist.generate_response(
                client_utterance, 
                state["dialogue_history"],
                state["is_wrapping_up"]
            )
            # print(f"Therapist results: {therapist_result}")
        
        # Update dialogue history
        if state["turn_count"] > 0:  # Not the first turn
            current_turn = {
                "turn": state["turn_count"],
                "client_utterance": client_utterance,  
                "therapist_utterance": therapist_result["therapist_utterance"],
                "client_mi_code": therapist_result.get("client_mi_code", ""),
                "therapist_mi_code": therapist_result.get("therapist_mi_code", ""),
                # "retrieved_contexts": retrieved_contexts
            }
            dialogue_history = state["dialogue_history"] + [current_turn]
        else:  # First turn
            current_turn = {
                "turn": state["turn_count"],
                "client_utterance": "",
                "therapist_utterance": therapist_result["therapist_utterance"],
                "client_mi_code": "",
                "therapist_mi_code": therapist_result.get("therapist_mi_code", "question"),
                # "retrieved_contexts": []
            }
            dialogue_history = [current_turn]
        
        # Update message history - Fix the append usage
        messages = state["messages"] + [f"Therapist:{therapist_result['therapist_utterance']}"]
        
        # Update state
        updated_state = {
            **state,
            "messages": messages,
            "dialogue_history": dialogue_history,
            "current_speaker": "client",
            "therapist_utterance": therapist_result["therapist_utterance"]
        }
        
        # Update current state
        self.current_state = updated_state
        
        return updated_state
    
    def _run_client_agent(self, state: AgentState) -> AgentState:
        """Run client agent"""
        # Get current therapist utterance
        therapist_utterance = state["therapist_utterance"]
        
        # Generate client response
        client_utterance = self.client.generate_response(
            therapist_utterance, 
            state["messages"],
        )
        
        # Update message history - Fix the append usage
        messages = state["messages"] + [f"Client:{client_utterance}"]
        
        # Increment turn counter
        turn_count = state["turn_count"] + 1
        
        # Update state
        updated_state = {
            **state,
            "messages": messages,
            "current_speaker": "therapist",
            "client_utterance": client_utterance,
            "turn_count": turn_count
        }
        
        # Update current state
        self.current_state = updated_state
        
        return updated_state
    

    #  modify t
    def _check_completion(self, state: AgentState) -> AgentState:
        """Check if the dialogue is complete"""
        # Check if maximum turns reached
        if state["turn_count"] >= self.max_turns:
            self.session_metadata["session_naturally_ended"] = False
            self.session_metadata["end_reason"] = f"Reached maximum turn limit ({self.max_turns})"
            print(f"Reached maximum turn limit ({self.max_turns}), ending conversation")
            updated_state = {
                **state,
                "is_complete": True
            }
            self.current_state = updated_state
            return updated_state
        
        # Signal wrap-up when approaching maximum turns
        if state["turn_count"] >= self.max_turns * 0.5:
        # if state["turn_count"] >= 5:
            print(f"Approaching maximum turns ({state['turn_count']}/{self.max_turns}), signaling wrap-up")
            state = {
                **state,
                "is_wrapping_up": True  # Set wrap-up flag
            }

        
        if state["turn_count"] > 10:
            # Prepare prompt
            completion_prompt = """
            You are a session monitor.
            TASK: Based on the conversation history, determine whether the counseling session should be ended.
            OUTPUT FORMAT: Return ONLY valid JSON (no markdown) with exactly the keys:          
            {
              "result": "<complete|continue>",
              "reason": "<brief one‑sentence justification>"
            }

            DECISION RULES (apply in order):  
            1. FAREWELL OVERRIDES:
            • Look at the therapist’s **last utterance that appears in the log**.
            • If it contains a clear closing / farewell cue such as
                "wrap up", "finish up", "as we close", "as we end", 
                "good-bye", "goodbye", "bye for now", "take care",
                "see you next time", "look forward to our next session",
                "talk to you soon", "until next time",
                or an explicit session summary + sign-off,
                AND it does **not** introduce a new topic that requires an answer **right now**,
                THEN → `"result": "complete"`.

            2. CLIENT ENDS:
            If the client explicitly says they wish to stop, leave, end, or offers a final thanks / good-bye after the therapist’s turn, → `"complete"`.

            3. OTHERWISE:
            If new topics emerge, the therapist asks a substantive question needing an immediate reply, or ending signals are missing/ambiguous, → `"continue"`.
            """
            
        # without adding any additional content.
            # Check if the session should end
            try:
                dialogue_history = []
                for msg in state["messages"][-10:]:
                    dialogue_history.append(msg)
                messages = [SystemMessage(content=completion_prompt),
                            HumanMessage(content=f"Dialogue history:{dialogue_history}")]

                
                parser = JsonOutputParser()
                # chain = self.completion_detector_llm | parser
                completion_check = self.completion_detector_llm.invoke(messages).content
                if getattr(self.completion_detector_llm, "model", "").lower().startswith("deepseek-r1"):
                    completion_check = re.sub(r"<think>.*?</think>\n?", "", completion_check, flags=re.DOTALL)
                completion_check = parser.invoke(completion_check)
                is_complete = completion_check["result"] == "complete"

                print("Completion check result: ", completion_check)
                
                if is_complete:
                    self.session_metadata["session_naturally_ended"] = True
                    self.session_metadata["end_reason"] = "Conversation naturally ended"
                    print("Detected natural end of conversation")
            except Exception as e:
                print(f"Completion detection error: {str(e)}")
                is_complete = False
            
            # Update state
            updated_state = {
                **state,
                "is_complete": is_complete
            }
            
            if state.get("is_wrapping_up", False):
                updated_state["is_wrapping_up"] = True
            # Update current state
            self.current_state = updated_state
            
            return updated_state
        else:
            return state
    
    def _should_complete(self, state: AgentState) -> bool:
        """Decide whether to continue the dialogue"""
        # Add additional safety check
        if state["turn_count"] >= self.max_turns:
            print(f"Safety check: Reached maximum turns {self.max_turns}, forcing dialogue to end")
            return True
        
        # Check if already completed
        is_complete = state.get("is_complete", False)
        
        # Print current status
        print(f"Turn {state['turn_count']}: is_complete = {is_complete}")
            
        return is_complete
    
    def _run_identifier(self, state: AgentState) -> AgentState:
        """identifier to identify the domains in the dialogue"""
        print("=======================Run Identifier=======================")
        
        try:
            if not state["dialogue_history"]:
                print("Warning: Dialogue history is empty, skipping domain identification")
                return {
                    **state,
                    "identified_domains": []
                }
            
            
            # use identifier to identify the domains
            identified_domains = self.identifier.identify_domains(
                state["dialogue_history"]
            )
            
            # update the state
            updated_state = {
                **state,
                "identified_domains": identified_domains
            }
            
            # update the current state
            self.current_state = updated_state
            
            return updated_state
        except Exception as e:
            print(f"Domain identification error: {str(e)}")
            print(f"Error details: {traceback.format_exc()}")
            # if error, return the original state and add empty identified results
            return {
                **state,
                "identified_domains": []
            }
    
    def _end_session(self, state: AgentState) -> AgentState:
        """End the session"""
        print(f"Session ended: {self.session_metadata['end_reason']}")
        
        # print the identified domains (if exists)
        if "identified_domains" in state and state["identified_domains"]:
            print("\nIdentified domains:")
            for domain in state["identified_domains"]:
                print(f"- {domain['domain']}: {domain['clue']}")
        
        # Update current state
        self.current_state = state
        return state
    
    ####################### Workflow Execution #########################
    def run_session(self, user_data: Dict[str, Any]) -> Dict:
        """Run the complete dialogue session
        
        Args:
            user_data: user data 
            
        Returns:
            A dictionary containing the dialogue history and identified domains
        """
        # Initialize client with user data
        self.client.process_user_data(user_data)
        
        # prepare the initial state with necessary fields
        initial_state = {
            "user_data": user_data,
            "messages": [],
            "dialogue_history": [],
            "current_speaker": "therapist",
            "is_complete": False,
            "client_utterance": "",
            "therapist_utterance": "",
            "turn_count": 0,
            "identified_domains": [],
            "is_wrapping_up": False  # Initialize wrap-up signal
        }
        
        # reset the current state
        self.current_state = None
        
        # execute the workflow
        start_time = time.time()
        try:
            print("========================Start executing the dialogue workflow========================")

            recursion_limit = max(10, self.max_turns * 4 + 5)
            print(f"Set recursion limit to: {recursion_limit}")
            
            final_state = self.workflow.invoke(
                initial_state, 
                config={"recursion_limit": recursion_limit}
            )
            print("=======================Dialogue workflow executed successfully=======================")
        except Exception as e:
            print(f"Workflow execution error: {str(e)}")
            # if error, return the current state
            if self.current_state and "dialogue_history" in self.current_state:
                print(f"Return partial dialogue history ({len(self.current_state['dialogue_history'])} turns)")
                
                # Run identifier even when an error occurs
                print("Running identifier to identify domains...")
                identified_domains = []
                try:
                    if self.current_state["dialogue_history"]:
                        identified_domains = self.identifier.identify_domains(
                            self.current_state["dialogue_history"]
                        )
                        if identified_domains:
                            print(f"Identified {len(identified_domains)} domains despite workflow error")
                        else:
                            print("No domains identified")
                except Exception as id_error:
                    print(f"Error in identifier during error recovery: {str(id_error)}")
                
                domains = {}
                try:
                    if "user_data" in self.current_state and isinstance(self.current_state["user_data"], dict):
                        # domains = self.current_state["user_data"].get("questionnaire", {}).get("level1", {}).get("result", {})
                        domains = self.current_state["user_data"].get("screening_results", {})
                except Exception as domain_error:
                    print(f"Error extracting domains: {str(domain_error)}")
                
                result = {
                    "dialogue_history": self.current_state["dialogue_history"],
                    "domains": domains,  
                    "identified_domains": identified_domains,
                    "session_metadata": {
                        **self.session_metadata,
                        "error": str(e),
                        "completed": False
                    }
                }
                return result
            else:
                print("Unable to get partial dialogue history, raise the original error")
                raise e
                
        end_time = time.time()
        
        # record the session duration
        duration = end_time - start_time
        # self.session_metadata["duration_seconds"] = duration
        self.session_metadata["completed"] = True
        print(f"========Dialogue completed, {len(final_state['dialogue_history'])} turns, {duration:.2f} seconds===============")
        
        # print("final_state: ", final_state)
        # return the dialogue history and identified domains
        result = {
            "dialogue_history": final_state["dialogue_history"],
            # "domains": final_state["user_data"]["questionnaire"]["level1"]["result"],
            "domains": final_state["user_data"]["screening_results"],
            "identified_domains": final_state.get("identified_domains", []),
            "session_metadata": self.session_metadata
        }
        return result




