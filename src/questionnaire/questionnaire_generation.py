import json
import os
import logging
from typing import Dict, Optional, List, Tuple
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings
from PyPDF2 import PdfReader
from pathlib import Path
from typing import List
from .schemas import Questionnaire, ClientResponse, Level1Result, Level2Result
import random
# LLMChain is not used - using new LangChain syntax (prompt | llm | parser) instead
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from .config import (
    QUESTIONNAIRE_DIR,
    DSM5_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    IDENTITY_TYPES,
    AGE_RANGES,
    DOMAIN_POOL,
    DOMAIN_ITEMS,
    QUESTIONNAIRE
)

class DocumentLoader:
    """Document loader class"""

    @staticmethod
    def load_pdf_documents(folder_path: str | Path) -> List[Document]:
        """Load PDF documents from a directory
        
        Args:
            folder_path: Directory path containing PDF files
            
        Returns:
            List of Document objects
        """
        documents = []
        folder_path = Path(folder_path).resolve()     
        
        if not folder_path.exists():
            raise FileNotFoundError(f"Directory not found: {folder_path}")
            
        pdf_files = list(folder_path.rglob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {folder_path} or its subdirectories")
            return documents
            
        for pdf_path in pdf_files:
            reader = PdfReader(str(pdf_path))
            text = ""
            try:
                for page_num, page in enumerate(reader.pages):            
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    
                    if text.strip():
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": str(pdf_path),
                                    # "filename": pdf_path.name,
                                    # "path": str(pdf_path),
                                    # "directory": str(pdf_path.parent)
                                }
                            )
                        ) 
            except Exception as e:
                print(f"Error loading PDF file: {pdf_path}")                     
        return documents
    
    @staticmethod
    def create_vector_store(documents, split_strategy) -> SKLearnVectorStore:
        """create vector store
        
        Args:
            documents: list of documents
            split_strategy: split strategy, "chunk" or "document"
            
        Returns:
            vector store object
        """
        if split_strategy == "chunk":
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            documents = text_splitter.split_documents(documents)
        elif split_strategy == "document":
            doc_splits = [doc for doc in documents]
            documents = doc_splits
            
        embeddings = NomicEmbeddings(
            model="nomic-embed-text-v1.5",
            inference_mode="local"
        )
        
        return SKLearnVectorStore.from_documents(
            documents=documents,
            embedding=embeddings
        ) 

class QuestionnaireProcessor:
    
    def __init__(self, llm):
        self.llm = llm
    
    def generate_user_profile(self, identity_type: str = None):
        """generate user profile
        
        Args:
            identity_type: identity type
            
        Returns:
            single line string containing age, identity and domain
        """
        identity = identity_type if identity_type else random.choice(IDENTITY_TYPES)
        age = random.randint(AGE_RANGES[identity][0], AGE_RANGES[identity][1])
        # domain = random.choice(DOMAIN_POOL)
        return f"{age}_{identity.lower()}"
    
    def extract_questionnaire(self, docs_txt: str) -> Tuple[Questionnaire, str]:
        """extract questionnaire content
        
        Args:
            docs_txt: document text
            
        Returns:
            questionnaire object and document text
        """
        parser = JsonOutputParser(pydantic_object=Questionnaire)
        format_instructions = parser.get_format_instructions()
        
        extract_prompt = PromptTemplate(
            template="""
            You are a mental health expert. Your task is to extract the questionnaire content from the following DSM-5-TR assessment document:
            {context}

            Instructions:
            1. Extract all assessment questions in their original order
            2. Note the answer range following each question:
                - If it is a scored question, indicate the score range (e.g., 0-4).
                - If it is a Yes/No question, indicate (Yes/No).
                - If there are other answer options, list them completely.
            {format_instructions}
            """,
            # 3. Map each question to ONE of these specific symptom domains accord to the document(Depression, Anger...)

            input_variables=["context"],
            partial_variables={"format_instructions": format_instructions}
        )
        
        chain = extract_prompt | self.llm | parser
        result = chain.invoke({"context": docs_txt})
        print("result:",result)
        return result, docs_txt
    
    def simulate_client_response(
        self,
        questionnaire: Questionnaire,
        client_info: str
    ) -> ClientResponse:
        """simulate client response
        
        Args:
            questionnaire: questionnaire object
            client_info: client information
            
        Returns:
            client response object
        """
        parser = JsonOutputParser(pydantic_object=ClientResponse)
        format_instructions = parser.get_format_instructions()
        
        client_prompt = PromptTemplate(
            template="""
            You are now a client seeking psychological counseling.
            
            Your basic information: {client_info}
                    
            Question list:
            {questions}
            
            YOUR TASK:
            For every single question above (exactly 23), you must:
            1.	Choose one integer score from 0, 1, 2, 3, or 4 that best fits the client’s feelings or behavior described in the question.
                - 0 = “Not at all / Never”
                - 4 = “Almost always / Extreme”
            2.	Write one short, conversational explanation (1-2 sentences) that reflects the severity implied by the chosen score, as if the client were speaking.

            Constraints:
            The length of the two arrays must equal the number of questions (here, 23). Do not skip or merge items.
            - `scores` must be a list of **23 integers**, with no extra text or comments.
            - `explanations` must be a list of **23 strings**, in the same order as the questions.
            - Do **NOT** include any question text, comments, or markdown in the output.
            - Do **NOT** include code fences or other non-JSON elements.

            OUTPUT FORMAT (JSON ONLY):
            Return nothing except valid JSON with exactly the two keys shown below.
            Keep the key names and the array order identical to the question order.
                {{
                “scores”: [s1, s2, …, s23],
                “explanations”: [“explanation1”, “explanation2”, …, “explanation23”]
                }}
            """,
            input_variables=["client_info", "questions"]
        )
        
        chain = client_prompt | self.llm | parser
        try:
            result = chain.invoke({
                "client_info": client_info,
                "questions": questionnaire["questions"]
            })
            return result
        except Exception as e:
            logging.error(f"Error in simulating client response: {e}")
            raise e

    

    def generate_diagnosis(
        self,
        client_response: ClientResponse,
        docs_txt: str,
        assessment_level: int = 1
    ) -> Level1Result | Level2Result:
        if assessment_level == 1:
            parser = JsonOutputParser(pydantic_object=Level1Result)
            template = """
            As a professional mental health expert, please analyze the client's responses based on 
            the DSM-5 documentation and provide the results:

            Screening document:
            {context}

            Client responses:
            {answers}

            Here is the mapping of answers to specific domain categories:
            {domain_mapping}

            Strictly analyze the client's responses based on the DSM-5 documentation and provide the results.
            
            Output format example:
            {{
                "Depression": {{
                    "score": ,
                    "result": "moderate",  // or "Yes"/"No"/"Don't know"
                }},
                "Anxiety": {{
                    "score": ,
                    "result":  
                }}
                // ... other domains
            }}
            
            {format_instructions}
            """

        else:
            parser = JsonOutputParser(pydantic_object=Level2Result)
            template = """
            As a professional mental health expert, please analyze the client's Level 2 responses:

            Assessment Document:
            {context}

            Client Responses:
            {answers}

            Please carefully calculate and provide the following information:
            1. Calculate the total raw score (raw_score): Sum the scores for all items.
            2. Calculate the converted score (processed_score):
               - If it is a T score, calculate it based on the conversion table provided in the assessment manual.
               - If it is another standardized score, please specify and calculate accordingly.
            3. Confirm the assessment domain (domain)
            4. Confirm the symptom severity (severity_level)

            {format_instructions}
            """
            
        format_instructions = parser.get_format_instructions()
        diagnosis_prompt = PromptTemplate(
            template=template,
            input_variables=["context", "answers","domain_mapping"],
            partial_variables={"format_instructions": format_instructions}
        )
        
        chain = diagnosis_prompt | self.llm | parser
        result = chain.invoke({
            "context": docs_txt,
            "answers": client_response["scores"],
            "domain_mapping": DOMAIN_ITEMS
        })
        
        return result 
    
    def calculate_domain_max_scores(self, client_response: ClientResponse, domain_mapping):
        domain_results = {}
        
        result_mapping = {
            0: "None",
            1: "Slight",
            2: "Mild",
            3: "Moderate",
            4: "Severe"
        }
        scores = client_response["scores"]
        for domain, indices in domain_mapping.items():
            domain_scores = [scores[i] for i in indices if i < len(scores)]        
            if domain_scores:
                max_score = max(domain_scores)
                result_text = result_mapping.get(max_score, "NA")
                
                domain_results[domain] = {
                    "domain_score": max_score,
                    "domain_result": result_text
                }
            else:
                domain_results[domain] = {
                    "domain_score": 0,
                    "domain_result": "No"
                }
        
        return domain_results

class QuestionnaireGenerator:
    """questionnaire agent class"""
    def __init__(self, llm):
        """initialize
        
        Args:
            llm: language model instance
        """
        self.llm = llm
        self.processor = QuestionnaireProcessor(llm)
        self.questionnaire_dir = QUESTIONNAIRE_DIR
        self.questionnaire_dir.mkdir(parents=True, exist_ok=True)
        self.data = {}
        
        # self.level1_retriever = self._init_retriever("Level1")
        # self.level2_retriever = self._init_retriever("Level2")

    def _init_retriever(self, level: str):
        """initialize specific level retriever"""

        level_dir = DSM5_DIR / level
        print(f"Initializing {level} retriever from: {level_dir}")
        
        
        try:
            if level_dir.exists():
                docs = DocumentLoader.load_pdf_documents(level_dir)
            else:
                raise FileNotFoundError(f"Directory not found: {level_dir}")
                
            vector_store = DocumentLoader.create_vector_store(docs, split_strategy="document")
            retriever = vector_store.as_retriever()
            print(f"Successfully created {level} retriever")
            return retriever
            
        except Exception as e:
            raise Exception(f"Error creating {level} retriever: {e}")
        
    def _get_user_json_path(self, user_id: str) -> Path:
        return self.questionnaire_dir / f"questionnaire_{user_id}.json"
            
    def _load_or_create_json(self):
        """load or create JSON file"""
        self.data = {}
            
    def _save_json(self, user_id: str = None):

        if user_id:
            user_json_path = self._get_user_json_path(user_id)
            with open(user_json_path, 'w', encoding='utf-8') as f:
                json.dump(self.data[user_id], f, indent=4, ensure_ascii=False)
        else:
            for uid in self.data:
                user_json_path = self._get_user_json_path(uid)
                with open(user_json_path, 'w', encoding='utf-8') as f:
                    json.dump(self.data[uid], f, indent=4, ensure_ascii=False)
            
    def generate_user(self, i, identity_type: Optional[str] = None) -> str:
        """generate new user
        
        Args:
            identity_type: identity type
            
        Returns:
            user ID
        """
        # existing_users = list(self.questionnaire_dir.glob("questionnaire_user*.json"))
        # next_user_num = len(existing_users) + 1
        
        # generate user ID
        user_id = f"user{i+1}"
        
        # generate user profile
        profile = self.processor.generate_user_profile(identity_type)
        
        # create user data structure
        self.data[user_id] = {
            "user_info": profile,
            "questionnaire": {
                "level1": {},
                # "level2": {}
            }
        }
        
        # save to JSON
        self._save_json(user_id)
        return user_id
    
    def complete_questionnaire(self, user_id: str) -> Dict:
        """complete questionnaire filling process
        
        Args:
            user_id: user ID
            
        Returns:
            questionnaire data
        """
        if user_id not in self.data:
            user_json_path = self._get_user_json_path(user_id)
            if user_json_path.exists():
                with open(user_json_path, 'r', encoding='utf-8') as f:
                    self.data[user_id] = json.load(f)
            else:
                raise ValueError(f"User {user_id} not found")
            
        user_info = self.data[user_id]["user_info"]
        
        # get related documents
        print("========= Starting Level 1 assessment =========")

        
        # level1_docs = self.level1_retriever.invoke(user_info, k=1)
        # level1_docs_txt = "\n\n".join(doc.page_content for doc in level1_docs)

        # using retriever to get related documents
        # Level 1 questionnaire
        # level1_questionnaire, level1_docs_txt = self.processor.extract_questionnaire(level1_docs_txt)

        # using preset documents
        level1_questionnaire = QUESTIONNAIRE
        level1_response = self.processor.simulate_client_response(level1_questionnaire, user_info)
        # level1_result = self.processor.generate_diagnosis(level1_response, docs_txt=level1_docs_txt, assessment_level=1)
        level1_result = self.processor.calculate_domain_max_scores(level1_response, DOMAIN_ITEMS)
        # save Level 1 data
        self.data[user_id]["questionnaire"]["level1"] = {
            "questionnaire": level1_questionnaire,
            "user_response": level1_response,
            "result": level1_result
        }
        self._save_json(user_id)      
        return self.data[user_id] 



