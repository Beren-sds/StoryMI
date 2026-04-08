from collections import OrderedDict
from typing import List, Dict, Optional, Any
import re
from sentence_transformers import SentenceTransformer, util
import evaluate
from nltk.util import ngrams
import os, sys
import pandas as pd
import json
from datetime import datetime

# Import shared components
from src.utils import initialize_llm, processed_json_file, processed_csv_file

# Import RAGAS for RAG evaluation
from ragas import evaluate as ragas_evaluate
from ragas.llms import LangchainLLMWrapper
from ragas import EvaluationDataset
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

class ConversationEvaluator:
    def __init__(self, local_llm=False):  # use gpt-5-nano by default for evaluation quality
        # Load BERT-based model for semantic similarity
        self.bert_model = SentenceTransformer("all-MiniLM-L6-v2")
        # Load evaluation metrics from Hugging Face's evaluate library
        # self.bleu = evaluate.load("bleu")
        # self.meteor = evaluate.load("meteor")
        # self.rouge = evaluate.load("rouge")
        
        # Use shared LLM component directly
        if local_llm:
            print("Using local LLM")
            self.llm = initialize_llm(local_llm=True, model_name="llama3.1:8b", temperature=0.5)
        else:
            print("Using remote LLM")
            # gpt-5-nano only supports temperature=1.0
            self.llm = initialize_llm(local_llm=False, model_name="gpt-5-nano", temperature=1.0)

        # Load evaluation rubrics
        rubrics_path = os.path.join(os.path.dirname(__file__), "evaluation_rubrics.json")
        with open(rubrics_path, "r") as f:
            self.rubrics = json.load(f)
        
        # Create results directory
        self.results_dir = "data/results/evaluation_results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def compute_ngram_diversity(self, responses, n=2):
        """Computes n-gram diversity score (bigram, trigram, etc.)
        Score	Interpretation
        0.0 - 0.3	Highly repetitive phrases
        0.3 - 0.6	Some phrase variation, but noticeable repetition
        0.6 - 0.9	Good diversity, minimal repeated phrases
        0.9 - 1.0	Very rich vocabulary, almost no repetition
        """
        ngram_list = []
        for response in responses:
            words = response.split()
            ngram_list.extend(list(ngrams(words, n)))  # Extract n-grams
        
        total_ngrams = len(ngram_list)
        unique_ngrams = len(set(ngram_list))
        
        return round(unique_ngrams / max(1, total_ngrams), 4)  # Avoid division by zero
    
    # def check_diversity(self, conversation):
    #     """Measures linguistic diversity based on unique sentence structures."""
    #     learner_responses = [line for line in conversation]
    #     unique_responses = set(learner_responses)
    #     return round(len(unique_responses) / max(1, len(learner_responses)), 4)
    
    # BLEU Score (n-gram precision-based metric)
    def compute_bleu(self, reference, generated):
        """Computes BLEU score using Hugging Face evaluate."""
        return round(self.bleu.compute(predictions=[generated], references=[reference])['bleu'], 4)

    # Instructional accuracy: ROUGE (useful for summarization-based tasks)
    def compute_rouge(self, reference, generated):
        """Computes ROUGE score using Hugging Face evaluate."""
        rouge_scores = self.rouge.compute(predictions=[generated], references=[reference])
        return {'ROUGE': round(rouge_scores['rouge1'], 4)}

    # METEOR Score (Recall-based metric, considers synonyms & stemming)
    def compute_meteor(self, reference, generated):
        """Computes METEOR score using Hugging Face evaluate."""
        return round(self.meteor.compute(predictions=[generated], references=[reference])['meteor'], 4)
        
    # Semantic similarity: BERTScore (context-aware similarity, recall-based matching)
    def compute_bert_score(self, reference, generated):
        """Computes semantic similarity using BERT embeddings."""
        ref_embedding = self.bert_model.encode(reference, convert_to_tensor=True)
        gen_embedding = self.bert_model.encode(generated, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(ref_embedding, gen_embedding).item()
        return round(similarity, 4)

    def compute_intent_accuracy(self, generated, reference):
        """Calculate the intent matching accuracy between generated and reference responses."""
        generated_intents_therapist = generated['therapist']
        reference_intents_therapist = reference['therapist']
        generated_intents_client = generated['client']
        reference_intents_client = reference['client']
        
        # Ensure both lists have the same length
        min_length = min(len(generated_intents_therapist), len(reference_intents_therapist))
        if min_length == 0:
            return 0.0, 0.0
            
        # Calculate intent matching accuracy
        correct_matches_therapist = sum(1 for g, r in zip(generated_intents_therapist, reference_intents_therapist) 
                                     if g == r or r == "therapist_input")
        correct_matches_client = sum(1 for g, r in zip(generated_intents_client, reference_intents_client) 
                                   if g == r)
        result = (correct_matches_client+correct_matches_therapist) / (len(generated_intents_client)+len(generated_intents_therapist))
        return round(result,4)
    
    def evaluate_with_ragas(self, reference, dialogue):
        # Extract queries (client utterances), responses (therapist utterances) and contexts
        queries = [turn['client_utterance'] for turn in dialogue]
        responses = [turn['therapist_utterance'] for turn in dialogue]
        
        # Ensure contexts are in the correct format - each context should be a list of strings
        contexts = []
        for turn in dialogue:
            docs = turn.get('relevant_docs', [])
            if docs is None:
                docs = []
            if isinstance(docs, str):
                docs = [docs]
            valid_docs = [doc for doc in docs if doc and isinstance(doc, str)]
            contexts.append(valid_docs)
        
        # Extract ground truths from reference
        ground_truths = [line for line in reference if line.startswith("Therapist:")]
        
        # Ensure all lists have the same length
        print(len(queries), len(responses), len(contexts), len(ground_truths))
        queries = queries[0:5]
        responses = responses[0:5]
        contexts = contexts[0:5]
        ground_truths = ground_truths[0:5]
        
        # Build dataset for RAGAS evaluation
        dataset = []
        for i, (query, response, context, ground_truth) in enumerate(zip(queries, responses, contexts, ground_truths)):
            dataset.append({
                "user_input": query,
                "retrieved_contexts": context,
                "response": response,
                "reference": ground_truth
            })
        # print("dataset: ", dataset)
        
        # Create evaluation dataset
        evaluation_dataset = EvaluationDataset.from_list(dataset)
        
        # Initialize evaluator LLM
        evaluator_llm = LangchainLLMWrapper(self.llm)
        
        # Run evaluation
        result = ragas_evaluate(
            dataset=evaluation_dataset,
            metrics=[
                LLMContextRecall(), 
                # Faithfulness(), 
                # FactualCorrectness()
            ],
            llm=evaluator_llm
        )
        return result

    def evaluate_with_reference(self, conversation, generated):
        reference = '\n'.join(generated)
        generated = '\n'.join([line for line in conversation if line.startswith("Therapist:")])
        # print("Generated: ", generated)
        # print("Reference: ", reference)
        rouge_scores = self.compute_rouge(reference, generated)
        # ref_llm_scores, response = self.check_factual_consistency(reference, generated)
        scores = {
            # **ref_llm_scores,
            'BLEU': self.compute_bleu(reference, generated),
            'METEOR': self.compute_meteor(reference, generated),
            'BERTScore': self.compute_bert_score(reference, generated),
            **rouge_scores
        }
        return scores
    
    def evaluate_with_llm(self, conversation):
        rubric_prompt = "\n".join([f"- {key} ({value['question']})" for key, value in self.rubrics.items() if self.rubrics[key]['reference'] == 'no'])
        format_prompt = ", ".join([f"{key}: X" for key in self.rubrics.keys() if self.rubrics[key]['reference'] == 'no'])
        prompt = f"""
        Evaluate the following therapist-client conversation:
        {conversation}      
        Score from 1 to 5:
        {rubric_prompt}
        Provide scores in this format: {format_prompt}
        """
        response = self.llm.invoke(prompt).content
        # self.logger.info(response)
        matches = re.findall(r'(\w+):\s*(\d+)\s*', response, re.IGNORECASE)
        scores = {}
        for key, value in matches:
            if key in self.rubrics.keys():
                scores[key.capitalize()] = int(value)
        return scores, response
    
    def evaluate(self, conversation:  List[str], reference: List[str]=None):
        # num_questions = self.calculate_question_ratio(conversation)
        # completed = self.check_completion(conversation)
        # diversity_score = self.check_diversity(conversation)
        ngram_diversity_score = self.compute_ngram_diversity(conversation)
        llm_scores, llm_scores_response = self.evaluate_with_llm(conversation)

        if reference:
            ref_scores = self.evaluate_with_reference(conversation, reference)
            return OrderedDict({
                "Diversity Score": round(ngram_diversity_score, 2),
                **llm_scores,
                **ref_scores
            })
        else:
            return OrderedDict({
                "Diversity Score": ngram_diversity_score,
                **llm_scores,
                "llm_scores_response": llm_scores_response,
            })
    
    
    def save_results(self, all_results, filename: str):
        """Save all results to a json file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        print(f"Evaluation results saved to: {filepath}")
    
    def main(self, start_index: int, end_index: int):
        all_results = {}
        sum_therapist = 0
        sum_client = 0
        
        if start_index >= end_index:
            print(f"Warning: start_index ({start_index}) should be less than end_index ({end_index})")
            return
            
        for i in range(start_index, end_index):
            conversation, intent_label, unformatted_dialogue = processed_json_file(i)
            # reference, reference_label = processed_csv_file(i)

            # print("unformatted_dialogue: ", unformatted_dialogue)
            # print("reference: ", reference)

            # ragas_scores = self.evaluate_with_ragas(dialogue=unformatted_dialogue, reference=reference)
            # print("ragas_scores: ", ragas_scores)
            # intent_accuracy = self.compute_intent_accuracy(intent_label)
            results = self.evaluate(conversation)
            print("results: ", results)
            # add to all results
            all_results[f"session_{i}"] = results
            # self.save_results(results, f"dialogue_session_{i}_by2llm.json")
        # save all results
        self.save_results(all_results, "all_results_by2llm.json")
        

if __name__ == "__main__":
    evaluator = ConversationEvaluator(local_llm=False)
    evaluator.main(0, 1)  
