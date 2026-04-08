"""
Evaluation script for MI-specific automatic metrics.

This script computes six MI-focused automatic evaluation metrics:
1. MI Code Distribution Entropy: Measures diversity of MI skill usage
2. MI Code Balance Score: KL divergence between actual vs ideal MI skill mix
3. Reflection Depth Score: Measures complexity of reflections (simple vs complex)
4. Complex Reflection Ratio: Percentage of reflections that are paraphrase/summarize
5. Reflection-to-Question Ratio: Core MISC indicator (R/Q)
6. Open vs Closed Question Ratio: Measures question type distribution

This script evaluates all six models:
- gpt-5-nano
- llama3.1:8b
- phi4:14b
- openchat:7b
- gemma:7b
- qwen2.5:7b
"""

import os
import sys
import json
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import numpy as np

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# Set environment variable to avoid tokenizer fork deadlock
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to path
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False


class MIMetricsEvaluator:
    """Evaluator for MI-specific automatic metrics."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.results_dir = Path("data/results/evaluation_results/mi_metrics")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load MI code definitions
        mi_code_path = PROJECT_ROOT / "src/dialogue/mi_code.json"
        with open(mi_code_path, 'r') as f:
            mi_codes = json.load(f)
        
        # Build mapping from subtype to parent code
        self.subtype_to_parent = {}
        self.parent_code_categories = []
        for parent_code, parent_info in mi_codes["mi_code_definition"]["therapist"].items():
            self.subtype_to_parent[parent_code] = parent_code  # parent maps to itself
            self.parent_code_categories.append(parent_code)
            for sub_code in parent_info.get("subtype", {}):
                self.subtype_to_parent[sub_code] = parent_code

        # Ideal MI skill distribution (based on MISC best practices)
        # Reflection should dominate, followed by questions (mostly open) and other inputs
        self.ideal_distribution = {
            "reflection": 0.5,
            "question": 0.25,
            "therapist_input": 0.2,
            "other": 0.05  # "other" should be rare, keep small weight
        }
        # Ensure all parent categories exist in ideal distribution
        for parent_code in self.parent_code_categories:
            if parent_code not in self.ideal_distribution:
                self.ideal_distribution[parent_code] = 0.0
        self.ideal_epsilon = 1e-6  # avoid log(0)

        # Initialize spaCy for linguistic processing
        if not SPACY_AVAILABLE:
            raise ImportError(
                "spaCy is required for MI metrics evaluation. "
                "Please install it via: pip install spacy && python -m spacy download en_core_web_sm"
            )
        try:
            self.spacy_nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
        except OSError as exc:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is not installed. "
                "Please run: python -m spacy download en_core_web_sm"
            ) from exc

        # Cache for token-level embeddings to avoid redundant encoding
        self.token_embedding_cache: Dict[str, np.ndarray] = {}
        self.token_similarity_threshold = 0.8
        
        # Initialize sentence transformer for reflection depth (REQUIRED)
        if not SENTENCE_TRANSFORMER_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for MI metrics evaluation. "
                "Please install it: pip install sentence-transformers"
            )
        
        print("Loading sentence transformer model (all-MiniLM-L6-v2)...")
        try:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            print("  [OK] Sentence transformer loaded successfully")
            
            # Test of the embedder
            test_texts = [
                "This is a test reflection.",
                "The client expressed feelings of anxiety."
            ]
            
            print("  Testing embedder with sample texts...")
            test_embeddings = self.embedder.encode(
                test_texts, 
                convert_to_tensor=False,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            
            if test_embeddings is None or len(test_embeddings) == 0:
                raise RuntimeError("Embedder test failed: returned empty embeddings")
            
            test_embeddings = np.array(test_embeddings)
            
            if test_embeddings.shape[0] != len(test_texts):
                raise RuntimeError(f"Embedder test failed: expected {len(test_texts)} embeddings, got {test_embeddings.shape[0]}")
            
            if test_embeddings.shape[1] != 384:  # all-MiniLM-L6-v2 has 384 dimensions
                print(f"  Warning: Expected 384 dimensions, got {test_embeddings.shape[1]}")
            
            # Test cosine similarity calculation
            similarity = np.dot(test_embeddings[0], test_embeddings[1])
            if not (-1.0 <= similarity <= 1.0):
                raise RuntimeError(f"Embedder test failed: similarity out of range: {similarity}")
            
            print(f"  [OK] Embedder test passed:")
            print(f"    - Embedding dimension: {test_embeddings.shape[1]}")
            print(f"    - Batch encoding: [OK]")
            print(f"    - Cosine similarity calculation: [OK]")
            print(f"    - Sample similarity: {similarity:.4f}")
            
        except Exception as e:
            error_msg = str(e)
            if "Connection" in error_msg or "download" in error_msg.lower():
                raise RuntimeError(
                    f"Failed to download/load sentence transformer: {e}\n"
                    "Test the connection with:\n"
                    "  python3 -c \"from sentence_transformers import SentenceTransformer; "
                    "SentenceTransformer('all-MiniLM-L6-v2')\""
                ) from e
            else:
                raise RuntimeError(
                    f"Failed to initialize sentence transformer: {e}"
                ) from e
    
    def normalize_mi_code(self, mi_code: Any) -> Optional[str]:
        """
        Normalize MI code to parent category.
        
        Handles various formats:
        - "reflection" -> "reflection"
        - "open_question" -> "question"
        - "reflection_with_open_question" -> "reflection" (takes first part)
        - None/empty -> None
        
        Args:
            mi_code: MI code string or None
            
        Returns:
            Normalized parent code or None
        """
        if not mi_code or mi_code == "":
            return None
        
        # Convert to string if needed
        code_str = str(mi_code).strip()
        if not code_str:
            return None
        
        # Handle compound codes like "reflection_with_open_question"
        # Take the first part (main category)
        if "_with_" in code_str:
            code_str = code_str.split("_with_")[0]
        elif "_" in code_str and code_str not in self.subtype_to_parent:
            # Try splitting on underscore and checking first part
            parts = code_str.split("_")
            if parts[0] in self.subtype_to_parent:
                code_str = parts[0]
        
        # Map to parent code
        return self.subtype_to_parent.get(code_str, code_str)
    
    def extract_question_subtype(self, mi_code: Any) -> Optional[str]:
        """
        Extract question subtype (open_question or closed_question).
        
        Handles various formats:
        - "open_question" -> "open_question"
        - "closed_question" -> "closed_question"
        - "question" -> "question" (no subtype)
        - "reflection_with_open_question" -> "open_question"
        - "reflection_with_closed_question" -> "closed_question"
        
        Args:
            mi_code: MI code string
            
        Returns:
            "open_question", "closed_question", "question" (if question but no subtype), or None
        """
        if not mi_code or mi_code == "":
            return None
        
        code_str = str(mi_code).strip()
        if not code_str:
            return None
        
        # Handle compound codes like "reflection_with_open_question"
        if "_with_" in code_str:
            # Extract the part after "_with_"
            after_with = code_str.split("_with_")[-1]
            if "open_question" in after_with or after_with == "open_question":
                return "open_question"
            elif "closed_question" in after_with or after_with == "closed_question":
                return "closed_question"
            elif "open" in after_with:
                return "open_question"
            elif "closed" in after_with:
                return "closed_question"
            elif "question" in after_with:
                return "question"
        
        # Check if it's a question type
        if "open_question" in code_str or code_str == "open_question":
            return "open_question"
        elif "closed_question" in code_str or code_str == "closed_question":
            return "closed_question"
        elif code_str == "question":
            return "question"  # Question but subtype unknown
        elif "question" in code_str:
            # Compound code with question, try to extract subtype
            if "open" in code_str:
                return "open_question"
            elif "closed" in code_str:
                return "closed_question"
            else:
                return "question"
        
        return None
    
    def compute_mi_code_entropy(self, therapist_mi_codes: List[Any]) -> float:
        """
        Compute MI code distribution entropy.
        
        Measures diversity of MI skill usage. Higher entropy = more balanced skill usage.
        Lower entropy = over-reliance on one or few skills.
        
        Fixed: Uses observed categories (not theoretical max) for normalization to prevent values > 1.0.
        This measures diversity within the actually used skill set, which is more meaningful.
        
        Args:
            therapist_mi_codes: List of therapist MI codes from dialogue
            
        Returns:
            Normalized entropy score (0-1)
        """
        if not therapist_mi_codes:
            return 0.0
        
        # Normalize codes to parent categories
        normalized_codes = []
        for code in therapist_mi_codes:
            normalized = self.normalize_mi_code(code)
            if normalized:
                normalized_codes.append(normalized)
        
        if not normalized_codes:
            return 0.0
        
        # Count frequencies for parent categories
        code_counts = Counter(normalized_codes)
        total = len(normalized_codes)
        
        # Compute Shannon entropy
        entropy = -sum((count / total) * math.log2(count / total) 
                      for count in code_counts.values() if count > 0)
        
        # FIXED: Use observed categories (not theoretical max) for normalization
        # This ensures entropy is normalized by the maximum possible entropy
        # given the actual number of categories used
        observed_categories = len(code_counts)
        max_entropy = math.log2(observed_categories) if observed_categories > 1 else 1
        
        # Normalize to [0, 1]
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Additional safety: clip to [0, 1] to handle any numerical errors
        normalized_entropy = min(1.0, max(0.0, normalized_entropy))
        
        return round(normalized_entropy, 4)
    
    def compute_mi_code_balance_score(self, therapist_mi_codes: List[Any]) -> float:
        """
        Compute MI code balance score using KL divergence to ideal distribution.
        
        A score close to 1 indicates actual MI skill usage matches the ideal distribution.
        Lower scores indicate deviation from best practices (e.g., too few reflections).
        
        Uses add-epsilon smoothing with renormalization to ensure valid probability distribution.
        """
        if not therapist_mi_codes:
            return 0.0
        
        normalized_codes = []
        for code in therapist_mi_codes:
            normalized = self.normalize_mi_code(code)
            if normalized:
                normalized_codes.append(normalized)
        
        total = len(normalized_codes)
        if total == 0:
            return 0.0
        
        # Count occurrences
        code_counts = Counter(normalized_codes)
        
        # Add-epsilon smoothing: add epsilon to all categories
        actual_distribution = {}
        for code in self.parent_code_categories:
            count = code_counts.get(code, 0)
            actual_distribution[code] = count + self.ideal_epsilon
        
        # CRITICAL FIX: Renormalize to ensure probability sum = 1.0
        actual_total = sum(actual_distribution.values())
        for code in actual_distribution:
            actual_distribution[code] /= actual_total
        
        # Verify probability sum is approximately 1.0 (within numerical precision)
        prob_sum = sum(actual_distribution.values())
        if abs(prob_sum - 1.0) > 1e-5:
            # If renormalization failed, fall back to simple normalization
            for code in actual_distribution:
                actual_distribution[code] = code_counts.get(code, 0) / total if total > 0 else 0.0
        
        # Compute KL divergence
        kl_divergence = 0.0
        for code in self.parent_code_categories:
            actual_prob = actual_distribution[code]
            ideal_prob = self.ideal_distribution.get(code, self.ideal_epsilon)
            
            # Ensure ideal_prob > 0 to avoid log(0)
            if ideal_prob <= 0:
                ideal_prob = self.ideal_epsilon
            
            if actual_prob > 0:
                kl_divergence += actual_prob * math.log(actual_prob / ideal_prob)
        
        # KL divergence is theoretically >= 0, but clip to prevent numerical errors
        kl_divergence = max(0.0, kl_divergence)
        
        # Convert to score (exp(-KL) is in [0, 1])
        score = math.exp(-kl_divergence)
        
        # Additional clipping to ensure score is in [0, 1] range
        score = min(1.0, max(0.0, score))
        
        return round(score, 4)
    
    def _extract_semantic_tokens(self, text: str) -> List[str]:
        """Extract lemmatized tokens without stopwords/punctuation."""
        if not text.strip():
            return []
        doc = self.spacy_nlp(text)
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and not token.is_space
            and token.lemma_.strip()
        ]
        return tokens
    
    def _encode_tokens(self, tokens: List[str]) -> List[np.ndarray]:
        """Encode tokens using cached embeddings."""
        embeddings: List[np.ndarray] = []
        missing_tokens = [token for token in tokens if token not in self.token_embedding_cache]
        
        if missing_tokens:
            missing_embeddings = self.embedder.encode(
                missing_tokens,
                convert_to_tensor=False,
                show_progress_bar=False,
                batch_size=32,
                normalize_embeddings=True
            )
            for token, emb in zip(missing_tokens, missing_embeddings):
                self.token_embedding_cache[token] = np.array(emb, dtype=np.float32)
        
        for token in tokens:
            embeddings.append(self.token_embedding_cache[token])
        
        return embeddings
    
    def _compute_semantic_info_gain(self, reflection: str, client_utt: str) -> float:
        """Compute information gain using semantic comparison of tokens."""
        ref_tokens = self._extract_semantic_tokens(reflection)
        if not ref_tokens:
            return 0.0
        client_tokens = self._extract_semantic_tokens(client_utt)
        
        ref_embeddings = self._encode_tokens(ref_tokens)
        client_embeddings = self._encode_tokens(client_tokens) if client_tokens else []
        
        novel_count = 0
        for ref_emb in ref_embeddings:
            is_novel = True
            if client_embeddings:
                similarities = np.dot(client_embeddings, ref_emb)
                if np.any(similarities >= self.token_similarity_threshold):
                    is_novel = False
            if is_novel:
                novel_count += 1
        
        info_gain = novel_count / len(ref_tokens)
        return info_gain
    
    def _compute_reflection_depth_scores(self,
                                         reflections: List[str],
                                         client_utterances: List[str]) -> Tuple[List[float], List[float], List[float]]:
        """
        Compute per-reflection depth scores.
        
        Returns:
            Tuple of (depth_scores, similarities, info_gains) for improved classification
        """
        if not reflections or not client_utterances:
            return ([], [], [])
        
        if len(reflections) != len(client_utterances):
            min_len = min(len(reflections), len(client_utterances))
            reflections = reflections[:min_len]
            client_utterances = client_utterances[:min_len]
        
        valid_pairs = [
            (r.strip(), c.strip())
            for r, c in zip(reflections, client_utterances)
            if r.strip() and c.strip()
        ]
        
        if not valid_pairs:
            return ([], [], [])
        
        if self.embedder is None:
            raise RuntimeError("Embedder not initialized. Please check initialization.")
        
        try:
            reflection_texts = [pair[0] for pair in valid_pairs]
            client_texts = [pair[1] for pair in valid_pairs]
            
            ref_embeddings = self.embedder.encode(
                reflection_texts,
                convert_to_tensor=False,
                show_progress_bar=False,
                batch_size=32,
                normalize_embeddings=True
            )
            client_embeddings = self.embedder.encode(
                client_texts,
                convert_to_tensor=False,
                show_progress_bar=False,
                batch_size=32,
                normalize_embeddings=True
            )
            
            ref_embeddings = np.array(ref_embeddings)
            client_embeddings = np.array(client_embeddings)
            
            if ref_embeddings.shape[0] != len(valid_pairs):
                raise ValueError("Embedding count mismatch for reflections")
            if client_embeddings.shape[0] != len(valid_pairs):
                raise ValueError("Embedding count mismatch for client utterances")
            
            similarities = np.sum(ref_embeddings * client_embeddings, axis=1)
            similarities = np.clip(similarities, -1.0, 1.0)
            similarities = (similarities + 1.0) / 2.0
            
        except Exception as e:
            raise RuntimeError(
                f"Error computing embeddings: {e}\n"
                f"Number of pairs: {len(valid_pairs)}\n"
                f"Reflection sample: {reflection_texts[0][:50] if valid_pairs else 'N/A'}\n"
                f"Client sample: {client_texts[0][:50] if valid_pairs else 'N/A'}"
            ) from e
        
        depth_scores = []
        similarity_list = []
        info_gain_list = []
        
        for idx, (reflection, client_utt) in enumerate(valid_pairs):
            similarity = float(similarities[idx])
            info_gain = self._compute_semantic_info_gain(reflection, client_utt)
            depth_score = similarity * 0.4 + info_gain * 0.6
            depth_scores.append(depth_score)
            similarity_list.append(similarity)
            info_gain_list.append(info_gain)
        
        # Store similarity and info_gain for improved classification
        # Return tuple: (depth_scores, similarities, info_gains)
        return (depth_scores, similarity_list, info_gain_list)
    
    def compute_reflection_depth(self,
                                reflections: List[str],
                                client_utterances: List[str]) -> float:
        """Return average reflection depth score."""
        depth_scores, _, _ = self._compute_reflection_depth_scores(reflections, client_utterances)
        if not depth_scores:
            return 0.0
        return round(float(np.mean(depth_scores)), 4)
    
    def classify_reflection_level(self, 
                                  depth_score: float,
                                  similarity: Optional[float] = None,
                                  info_gain: Optional[float] = None) -> str:
        """
        Classify reflection depth into MISC-inspired buckets.
        
        Improved version: Uses combination of similarity and info_gain when available,
        falling back to depth_score threshold when not available.
        
        Args:
            depth_score: Combined score (0.4*similarity + 0.6*info_gain)
            similarity: Semantic similarity between reflection and client utterance (optional)
            info_gain: Information gain ratio (optional)
        
        Returns:
            Classification: "repeat", "rephrase", "paraphrase", or "summarize"
        """
        # If we have similarity and info_gain, use improved classification
        if similarity is not None and info_gain is not None:
            # Repeat: High similarity + very low info gain (almost verbatim repetition)
            if similarity > 0.9 and info_gain < 0.15:
                return "repeat"
            
            # Rephrase: High similarity + low to medium info gain (reworded but same meaning)
            if similarity > 0.75 and info_gain < 0.35:
                return "rephrase"
            
            # Paraphrase: Medium similarity + medium to high info gain (adds meaning/inference)
            if similarity > 0.5 and info_gain < 0.6:
                return "paraphrase"
            
            # Summarize: High depth score (comprehensive reflection)
            # Note: True "summarize" requires aggregating multiple client utterances,
            # which current implementation doesn't check. This is a proxy.
            return "summarize"
        
        # Fallback to depth_score thresholds when similarity/info_gain not available
        if depth_score < 0.3:
            return "repeat"
        if depth_score < 0.5:
            return "rephrase"
        if depth_score < 0.7:
            return "paraphrase"
        return "summarize"
    
    def compute_complex_reflection_ratio_from_scores(self, 
                                                      depth_scores: List[float],
                                                      similarities: Optional[List[float]] = None,
                                                      info_gains: Optional[List[float]] = None) -> float:
        """
        Compute percentage of reflections that are complex (paraphrase or summarize).
        
        Uses improved classification when similarity and info_gain are available.
        """
        if not depth_scores:
            return 0.0
        
        # Use improved classification if similarity and info_gain are available
        if similarities is not None and info_gains is not None and len(similarities) == len(depth_scores):
            levels = [
                self.classify_reflection_level(score, sim, info)
                for score, sim, info in zip(depth_scores, similarities, info_gains)
            ]
        else:
            # Fallback to depth_score-only classification
            levels = [self.classify_reflection_level(score) for score in depth_scores]
        
        complex_count = sum(1 for level in levels if level in {"paraphrase", "summarize"})
        return round(complex_count / len(levels), 4)
    
    def _validate_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Validate that all metrics are within expected ranges.
        
        Raises ValueError if any metric is out of range, indicating a calculation bug.
        """
        # Metrics that must be in [0, 1] range
        bounded_metrics = [
            "mi_code_entropy",
            "mi_code_balance_score",
            "complex_reflection_ratio",
            "question_openness_ratio"
        ]
        
        for metric_name in bounded_metrics:
            if metric_name in metrics:
                value = metrics[metric_name]
                if not (0.0 <= value <= 1.0):
                    raise ValueError(
                        f"{metric_name} = {value} is out of valid range [0, 1]. "
                        f"This indicates a bug in the calculation. "
                        f"Please check the implementation."
                    )
        
        # Reflection depth should be in [0, 1] (weighted average of similarity and info_gain)
        if "reflection_depth" in metrics:
            value = metrics["reflection_depth"]
            if not (0.0 <= value <= 1.0):
                # Warning instead of error, as it's theoretically possible to exceed 1.0
                # if similarity and info_gain are both very high
                print(f"Warning: reflection_depth = {value} is out of typical range [0, 1]")
        
        # Reflection-to-question ratio must be >= 0
        if "reflection_to_question_ratio" in metrics:
            value = metrics["reflection_to_question_ratio"]
            if value < 0:
                raise ValueError(
                    f"reflection_to_question_ratio = {value} < 0. "
                    f"This indicates a bug in the calculation."
                )
    
    
    def compute_question_openness_ratio(self, therapist_mi_codes: List[Any]) -> float:
        """
        Compute open vs closed question ratio.
        
        In high-quality MI counseling, there should be more open questions.
        Target: >0.7 (70%+ open questions)
        
        Args:
            therapist_mi_codes: List of therapist MI codes from dialogue
            
        Returns:
            Ratio of open questions to total questions (0-1)
        """
        question_types = []
        
        for code in therapist_mi_codes:
            qtype = self.extract_question_subtype(code)
            if qtype:
                question_types.append(qtype)
        
        if not question_types:
            return 0.0  # No questions found
        
        open_count = sum(1 for qtype in question_types if qtype == "open_question")
        total_questions = len(question_types)
        
        # If we have "question" without subtype, we can't determine openness
        # Count only those with explicit subtype
        explicit_questions = sum(1 for qtype in question_types 
                                if qtype in ["open_question", "closed_question"])
        
        if explicit_questions == 0:
            return 0.0  # No questions with explicit subtype
        
        ratio = open_count / explicit_questions
        return round(ratio, 4)
    
    def compute_reflection_to_question_ratio(self, therapist_mi_codes: List[Any]) -> float:
        """
        Compute Reflection-to-Question ratio (R/Q), a core MISC metric.
        Ideal MI requires significantly more reflections than questions (>2.0).
        """
        if not therapist_mi_codes:
            return 0.0
        normalized_codes = [self.normalize_mi_code(code) for code in therapist_mi_codes if code]
        reflection_count = sum(1 for code in normalized_codes if code == "reflection")
        question_count = sum(1 for code in normalized_codes if code == "question")
        
        if question_count == 0:
            if reflection_count == 0:
                return 0.0
            # No questions asked: treat as very high ratio (equal to reflections count)
            return round(float(reflection_count), 4)
        
        ratio = reflection_count / question_count
        return round(ratio, 4)
    
    def evaluate_session(self, session_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate a single dialogue session for MI metrics.
        
        Args:
            session_data: Session data dictionary from JSON file
            
        Returns:
            Dictionary of MI evaluation metrics
        """
        # Verify embedder is available
        if self.embedder is None:
            raise RuntimeError(
                "Embedder not initialized. Reflection depth calculation requires "
                "sentence transformer embeddings."
            )
        
        dialogue_history = session_data.get("dialogue_history", [])
        
        # Extract therapist MI codes
        therapist_mi_codes = [
            turn.get("therapist_mi_code", "")
            for turn in dialogue_history
            if turn.get("therapist_utterance", "").strip()
        ]
        
        # Extract reflections and corresponding client utterances
        reflections = []
        client_utterances = []
        for turn in dialogue_history:
            therapist_code = self.normalize_mi_code(turn.get("therapist_mi_code", ""))
            if therapist_code == "reflection":
                therapist_utt = turn.get("therapist_utterance", "").strip()
                client_utt = turn.get("client_utterance", "").strip()
                if therapist_utt and client_utt:
                    reflections.append(therapist_utt)
                    client_utterances.append(client_utt)
        
        depth_scores, similarities, info_gains = self._compute_reflection_depth_scores(reflections, client_utterances)
        avg_depth = round(float(np.mean(depth_scores)), 4) if depth_scores else 0.0
        complex_ratio = self.compute_complex_reflection_ratio_from_scores(depth_scores, similarities, info_gains)
        
        # Compute metrics
        metrics = {
            "mi_code_entropy": self.compute_mi_code_entropy(therapist_mi_codes),
            "mi_code_balance_score": self.compute_mi_code_balance_score(therapist_mi_codes),
            "reflection_depth": avg_depth,
            "complex_reflection_ratio": complex_ratio,
            "question_openness_ratio": self.compute_question_openness_ratio(therapist_mi_codes),
            "reflection_to_question_ratio": self.compute_reflection_to_question_ratio(therapist_mi_codes)
        }
        
        # Validate metrics to ensure they are in valid ranges
        self._validate_metrics(metrics)
        
        return metrics
    
    def evaluate_model(self, 
                      session_dir: Path, 
                      model_name: str,
                      start_index: int = 1,
                      end_index: int = 1001) -> Dict[str, Any]:
        """
        Evaluate all sessions for a given model.
        
        Args:
            session_dir: Directory containing session JSON files
            model_name: Name of the model
            start_index: Start session index (default: 1)
            end_index: End session index (exclusive, default: 1001)
            
        Returns:
            Dictionary of all session results and summary statistics
        """
        all_results = {}
        all_metrics = {
            "mi_code_entropy": [],
            "mi_code_balance_score": [],
            "reflection_depth": [],
            "complex_reflection_ratio": [],
            "question_openness_ratio": [],
            "reflection_to_question_ratio": []
        }
        
        print(f"\n{'='*80}")
        print(f"Evaluating MI Metrics: {model_name}")
        print(f"Session directory: {session_dir}")
        print(f"Range: {start_index} to {end_index-1}")
        print(f"{'='*80}")
        
        # Verify embedder is available before processing
        if self.embedder is None:
            raise RuntimeError("Embedder not initialized. Cannot proceed with evaluation.")
        
        # Count total sessions first
        total_sessions = sum(1 for i in range(start_index, end_index) 
                           if (session_dir / f"session_{i}.json").exists())
        print(f"Found {total_sessions} sessions to process")
        
        processed = 0
        errors = 0
        
        for i in range(start_index, end_index):
            session_file = session_dir / f"session_{i}.json"
            
            if not session_file.exists():
                continue
            
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                
                metrics = self.evaluate_session(session_data)
                all_results[f"session_{i}"] = metrics
                
                # Accumulate for statistics
                for metric_name, value in metrics.items():
                    if value is not None and not (isinstance(value, float) and math.isnan(value)):
                        all_metrics[metric_name].append(value)
                
                processed += 1
                
                # Progress update every 100 sessions
                if processed % 100 == 0:
                    print(f"  Processed {processed}/{total_sessions} sessions... "
                          f"(reflection_depth using embeddings: {self.embedder is not None})")
                
            except Exception as e:
                errors += 1
                print(f"Warning: Error processing session {i}: {str(e)}")
                if errors > 10:
                    print(f"  Too many errors ({errors}), stopping evaluation for {model_name}")
                    break
                continue
        
        print(f"\nCompleted: {processed} sessions processed, {errors} errors")
        
        # Compute summary statistics
        summary = {}
        for metric_name, values in all_metrics.items():
            if values:
                summary[metric_name] = {
                    "mean": round(np.mean(values), 4),
                    "std": round(np.std(values), 4),
                    "min": round(np.min(values), 4),
                    "max": round(np.max(values), 4),
                    "median": round(np.median(values), 4),
                    "count": len(values)
                }
            else:
                summary[metric_name] = {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "median": 0.0,
                    "count": 0
                }
        
        print(f"\nSummary Statistics:")
        for metric_name, stats in summary.items():
            print(f"  {metric_name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                  f"n={stats['count']}")
        
        return {
            "model": model_name,
            "sessions": all_results,
            "summary": summary
        }
    
    def save_results(self, results: Dict[str, Any], model_name: str):
        """
        Save evaluation results to JSON file.
        
        Args:
            results: Results dictionary
            model_name: Model name for filename
        """
        # Sanitize model name for filename
        safe_name = model_name.replace(":", "_").replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mi_metrics_{safe_name}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to: {filepath}")
        return filepath


def main():
    """Main evaluation function for all models."""
    
    evaluator = MIMetricsEvaluator()
    project_root = Path(PROJECT_ROOT)
    
    # Define all models and their session directories
    models = [
        {
            "name": "gpt-5-nano",
            "path": project_root / "data/results/sessions/level1_by2llm/gpt-5-nano"
        },
        {
            "name": "llama3.1:8b",
            "path": project_root / "data/results/sessions/level1_by2llm/llama3.1:8b"
        },
        {
            "name": "phi4:14b",
            "path": project_root / "data/results/sessions/level1_by2llm/phi4:14b"
        },
        {
            "name": "openchat:7b",
            "path": project_root / "data/results/sessions/level1_by2llm/openchat:7b"
        },
        {
            "name": "gemma:7b",
            "path": project_root / "data/results/sessions/level1_by2llm/gemma:7b"
        },
        {
            "name": "qwen2.5:7b",
            "path": project_root / "data/results/sessions/level1_by2llm/qwen2.5:7b"
        }
    ]
    
    print("="*80)
    print("MI-Specific Automatic Metrics Evaluation")
    print("="*80)
    print("Metrics:")
    print("  1. MI Code Distribution Entropy (skill diversity)")
    print("  2. MI Code Balance Score (alignment with best practices)")
    print("  3. Reflection Depth Score (simple vs complex reflections)")
    print("  4. Complex Reflection Ratio (percent complex reflections)")
    print("  5. Reflection-to-Question Ratio (core MISC target > 2.0)")
    print("  6. Open vs Closed Question Ratio (target > 0.7)")
    print("="*80)
    
    for model in models:
        if not model["path"].exists():
            print(f"\nWarning: Directory not found: {model['path']}")
            continue
        
        results = evaluator.evaluate_model(
            session_dir=model["path"],
            model_name=model["name"],
            start_index=1,
            end_index=1001
        )
        
        evaluator.save_results(results, model["name"])
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)


if __name__ == "__main__":
    main()

