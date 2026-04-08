"""Data Model Definitions"""
from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional

class Questionnaire(BaseModel):
    """Questionnaire Model"""
    questions: List[str] = Field(description="List of assessment questions and answer ranges")
    # domains: Dict[str, List[int]] = Field(description="Domain classifications with corresponding question indices")

class ClientResponse(BaseModel):
    """Client Response Model"""
    scores: List[int] = Field(description="List of scores for each question")
    explanations: List[str] = Field(description="Conversational explanations for each answer")

class DomainResult(BaseModel):
    """Domain Result Model"""
    domain_score: int = Field(description="Domain raw score")
    domain_result: Union[str, float] = Field(description="Domain result")
    # suggested_measures: List[str] = Field(description="Recommended Level 2 assessment measures", default_factory=list)

class Level1Result(BaseModel):
    """Level 1 Assessment Result Model"""
    domains: Dict[str, DomainResult] = Field(description="Results and recommendations for each domain")

class Level2Result(BaseModel):
    """Level 2 Assessment Result Model"""
    raw_score: int = Field(description="Total raw score")
    processed_score: float = Field(description="Converted score (e.g., T score)")
    severity_level: str = Field(description="Symptom severity")
    domain: str = Field(description="Assessment domain")

class QuestionnaireData(BaseModel):
    """Questionnaire Data Model"""
    user_info: str = Field(..., description="Basic user information")
    questionnaire: Dict = Field(default_factory=dict, description="Questionnaire data")