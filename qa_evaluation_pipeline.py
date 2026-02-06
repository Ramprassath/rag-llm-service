# qa_evaluation_pipeline_FIXED.py
# This file contains ALL critical fixes applied
# Changes marked with # ✅ FIX

import torch
import os
import json
import logging
import pickle
import pandas as pd
import numpy as np
import faiss
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import warnings
import re
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Data structure for query results"""
    question: str
    question_id: int
    retrieved_documents: List[Dict]
    top_k_used: int
    retrieval_method: str
    bm25_weight: float
    vector_weight: float
    detected_law_type: Optional[str] = None
    query_embedding: Optional[List[float]] = None
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result.pop('query_embedding', None)
        return result


@dataclass
class EvaluationMetrics:
    """Data structure for evaluation metrics"""
    total_questions: int
    successful_retrievals: int
    failed_retrievals: int
    avg_top1_score: float
    avg_top5_score: float
    avg_top10_score: float
    avg_retrieval_time_ms: float
    coverage_by_law_type: Dict[str, int]
    score_distribution: Dict[str, int]
    hybrid_stats: Dict[str, float]
    law_type_detection_stats: Dict[str, int]
    timestamp: str


class QAEvaluationPipeline:
    """Pipeline for REAL Hybrid BM25 + Vector retrieval - FIXED VERSION"""
    
    def __init__(self,
                 qa_csv_path: str,
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 index_path: str = "./faiss_index",
                 metadata_path: str = "./faiss_metadata.pkl",
                 output_dir: str = "./evaluation_results",
                 gpu_id: int = None,
                 use_cpu: bool = False,
                 use_hybrid: bool = True,
                 default_bm25_weight: float = 0.6,
                 default_vector_weight: float = 0.4,
                 enable_law_type_filtering: bool = True):
        
        self.qa_csv_path = qa_csv_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.use_hybrid = use_hybrid
        self.default_bm25_weight = default_bm25_weight
        self.default_vector_weight = default_vector_weight
        self.enable_law_type_filtering = enable_law_type_filtering
        
        logger.info("="*80)
        logger.info("Initializing FIXED Hybrid BM25 + Vector Retrieval Pipeline")
        logger.info("="*80)
        logger.info(f"Hybrid Mode: {'ENABLED' if use_hybrid else 'DISABLED (Vector Only)'}")
        logger.info(f"Default Weights - BM25: {default_bm25_weight}, Vector: {default_vector_weight}")
        logger.info(f"Law Type Filtering: {'ENABLED' if enable_law_type_filtering else 'DISABLED'}")
        
        # GPU Selection
        self.device = self._select_device(gpu_id, use_cpu)
        logger.info(f"Using device: {self.device}")
        
        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name, device=self.device)
        
        # Load FAISS index
        logger.info(f"Loading FAISS index from: {index_path}")
        self.index = faiss.read_index(index_path)
        
        # Check FAISS index type
        self._check_faiss_index_type()
        
        # Load metadata
        logger.info(f"Loading metadata from: {metadata_path}")
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        logger.info(f"✅ Loaded: {self.index.ntotal} vectors, {len(self.metadata)} documents")
        
        # ✅ FIX #1: Build CANONICAL law type index
        if self.enable_law_type_filtering:
            self._build_law_type_index()
        
        # Initialize BM25 if hybrid mode is enabled
        if self.use_hybrid:
            logger.info("\n" + "="*80)
            logger.info("Building BM25 Index...")
            logger.info("="*80)
            self.bm25_index = self._initialize_bm25()
            logger.info(f"✅ BM25 index ready with {len(self.bm25_corpus)} documents")
            logger.info("="*80 + "\n")
        else:
            self.bm25_index = None
            self.bm25_corpus = None
        
        # Load QA dataset
        self.qa_data = self._load_qa_dataset()
    
    # ✅ FIX #1: NEW METHOD - Canonical Law Family Assignment
    def _normalize_law_family(self, doc: Dict) -> str:
        """
        Assign canonical law family based on multiple signals.
        Returns: 'constitution' | 'criminal' | 'civil' | 'other'
        """
        # Check source file first (most reliable)
        source_file = doc.get('source_file', '').lower()
        
        if 'constitution' in source_file or 'coi' in source_file:
            return 'constitution'
        
        if any(x in source_file for x in ['bns', 'ipc', 'crpc', 'bnss']):
            return 'criminal'
        
        if 'cpc' in source_file or 'civil' in source_file:
            return 'civil'
        
        # Check law_type field (secondary)
        law_type = doc.get('law_type', '').lower()
        
        if 'constitution' in law_type:
            return 'constitution'
        
        if any(x in law_type for x in ['penal', 'criminal', 'ipc', 'crpc', 'bns', 'bnss', 'bharatiya nyaya sanhita']):
            return 'criminal'
        
        if any(x in law_type for x in ['civil', 'cpc']):
            return 'civil'
        
        # Check for article numbers (Constitution-specific)
        if 'article_number' in doc or 'part_number' in doc:
            if 'section_number' not in doc:  # Not a regular act
                return 'constitution'
        
        return 'other'
    
    # ✅ FIX #1: REPLACED METHOD - Canonical Law Family Index
    def _build_law_type_index(self):
        """Build CANONICAL law family index with proper tagging"""
        logger.info("Building canonical law family index...")
        
        self.law_type_map = {
            'constitution': [],
            'criminal': [],
            'civil': [],
            'other': []
        }
        
        # Add law_family to each document
        for idx, doc in enumerate(self.metadata):
            law_family = self._normalize_law_family(doc)
            doc['law_family'] = law_family  # ✅ CRITICAL: Store in metadata
            self.law_type_map[law_family].append(idx)
        
        logger.info("Law family distribution:")
        for family, indices in self.law_type_map.items():
            logger.info(f"  {family}: {len(indices)} documents")
        
        # ✅ VALIDATION CHECK
        if len(self.law_type_map['constitution']) == 0:
            logger.error("❌ NO CONSTITUTION DOCUMENTS FOUND! Check source files.")
        else:
            logger.info(f"✅ Constitution docs: {len(self.law_type_map['constitution'])}")
    
    # ✅ FIX #2: REPLACED METHOD - Aligned Law Type Detection
    def _detect_law_type_and_reference(self, question: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Detect law FAMILY and specific reference from question.
        Returns: (law_family, reference_type)
        
        Maps to: 'constitution' | 'criminal' | 'civil' | None
        """
        q_lower = question.lower()
        
        # Article → Constitution (HIGH CONFIDENCE)
        if re.search(r'\barticle\s+\d+', q_lower):
            return 'constitution', 'article'
        
        # Constitution keywords
        if any(kw in q_lower for kw in ['constitution', 'fundamental right', 
                                          'directive principle', 'preamble']):
            return 'constitution', None
        
        # Section + Criminal law
        if re.search(r'\bsection\s+\d+', q_lower):
            if any(kw in q_lower for kw in ['ipc', 'indian penal code', 'bns', 
                                              'bharatiya nyaya sanhita', 'crpc', 
                                              'criminal procedure', 'bnss']):
                return 'criminal', 'section'
            
            if any(kw in q_lower for kw in ['cpc', 'civil procedure']):
                return 'civil', 'section'
            
            # Generic section → could be any, don't filter
            return None, 'section'
        
        # Criminal law keywords (no section)
        if any(kw in q_lower for kw in ['ipc', 'penal code', 'bns', 'crpc', 
                                          'criminal', 'offense', 'punishment']):
            return 'criminal', None
        
        # Civil law keywords
        if any(kw in q_lower for kw in ['cpc', 'civil procedure', 'suit', 
                                          'plaintiff', 'decree']):
            return 'civil', None
        
        return None, None
    
    # ✅ FIX #6: NEW METHOD - Extract Reference Numbers
    def _extract_reference_number(self, question: str, reference_type: Optional[str]) -> Optional[int]:
        """Extract article or section number from question"""
        if not reference_type:
            return None
        
        q_lower = question.lower()
        
        if reference_type == 'article':
            match = re.search(r'\barticle\s+(\d+)', q_lower)
            if match:
                return int(match.group(1))
        
        elif 'section' in reference_type:
            match = re.search(r'\bsection\s+(\d+)', q_lower)
            if match:
                return int(match.group(1))
        
        return None
    
    # ✅ FIX #6: NEW METHOD - Boost Exact Matches
    def _boost_exact_reference_match(self, hybrid_scores: Dict, 
                                      reference_number: Optional[int],
                                      reference_type: Optional[str]):
        """Boost scores for exact article/section matches"""
        if reference_number is None or reference_type is None:
            return
        
        boosted_count = 0
        for idx, scores in hybrid_scores.items():
            doc = self.metadata[idx]
            
            # Check for exact match
            if reference_type == 'article':
                doc_number = doc.get('article_number')
                if doc_number:
                    try:
                        if int(doc_number) == reference_number:
                            scores['hybrid_score'] *= 1.5  # 50% boost
                            boosted_count += 1
                    except (ValueError, TypeError):
                        pass
            
            elif 'section' in reference_type:
                doc_number = doc.get('section_number')
                if doc_number:
                    try:
                        if int(doc_number) == reference_number:
                            scores['hybrid_score'] *= 1.5
                            boosted_count += 1
                    except (ValueError, TypeError):
                        pass
        
        if boosted_count > 0:
            logger.debug(f"Boosted {boosted_count} docs for exact {reference_type} {reference_number} match")
    
    def _check_faiss_index_type(self):
        """Check FAISS index type and warn about score interpretation"""
        index_type = type(self.index).__name__
        logger.info(f"\nFAISS Index Type: {index_type}")
        
        if "IP" in index_type or "Flat" in index_type:
            self.faiss_is_similarity = True
            logger.info("✅ Index returns SIMILARITY scores (higher = better)")
        elif "L2" in index_type:
            self.faiss_is_similarity = False
            logger.warning("⚠️  Index returns DISTANCE scores (lower = better)")
            logger.warning("    Will convert: similarity = 1 / (1 + distance)")
        else:
            self.faiss_is_similarity = True
            logger.warning(f"⚠️  Unknown index type: {index_type}, assuming similarity")
        
        logger.info("")
    
    def _select_device(self, gpu_id: int = None, use_cpu: bool = False) -> str:
        """Select the appropriate device"""
        if use_cpu:
            return "cpu"
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU")
            return "cpu"
        
        num_gpus = torch.cuda.device_count()
        logger.info(f"Found {num_gpus} GPU(s) available")
        
        if gpu_id is not None:
            if gpu_id < 0 or gpu_id >= num_gpus:
                raise ValueError(f"GPU {gpu_id} not available")
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            logger.info(f"Using GPU {gpu_id}")
            return "cuda:0"
        else:
            best_gpu = 0
            max_free = 0
            for i in range(num_gpus):
                total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                free = total_mem - allocated
                if free > max_free:
                    max_free = free
                    best_gpu = i
            
            os.environ['CUDA_VISIBLE_DEVICES'] = str(best_gpu)
            logger.info(f"Auto-selected GPU {best_gpu} with {max_free:.1f}GB free")
            return "cuda:0"
    
    def _initialize_bm25(self) -> BM25Okapi:
        """Initialize BM25 index from metadata"""
        self.bm25_corpus = []
        self.bm25_doc_mapping = []
        
        for idx, doc in enumerate(tqdm(self.metadata, desc="Building BM25 corpus")):
            text_parts = []
            
            # Law identification
            if 'law_type' in doc:
                text_parts.append(doc['law_type'])
            
            # Section/Article identification
            if 'section_number' in doc:
                text_parts.append(f"section {doc['section_number']}")
            if 'article_number' in doc:
                text_parts.append(f"article {doc['article_number']}")
            
            # Main content
            if 'section_text' in doc:
                text_parts.append(doc['section_text'])
            elif 'text' in doc:
                text_parts.append(doc['text'])
            elif 'content' in doc:
                text_parts.append(doc['content'])
            
            # Description
            if 'section_description' in doc:
                text_parts.append(doc['section_description'])
            
            combined_text = " ".join(text_parts)
            tokenized = self._tokenize_for_bm25(combined_text)
            self.bm25_corpus.append(tokenized)
            self.bm25_doc_mapping.append(idx)
        
        return BM25Okapi(self.bm25_corpus)
    
    def _tokenize_for_bm25(self, text: str) -> List[str]:
        """Simple word-based tokenization for BM25"""
        tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    def _load_qa_dataset(self) -> pd.DataFrame:
        """Load and validate QA dataset"""
        logger.info(f"Loading QA dataset from: {self.qa_csv_path}")
        
        try:
            df = pd.read_csv(self.qa_csv_path)
            
            if 'QA' not in df.columns and len(df.columns) > 0:
                df.columns = ['QA']
            
            df = df.dropna(subset=['QA'])
            df = df[df['QA'].str.strip() != '']
            df = df.reset_index(drop=True)
            df['question_id'] = df.index
            
            logger.info(f"✅ Loaded {len(df)} questions from dataset")
            logger.info("Sample questions:")
            for i in range(min(3, len(df))):
                logger.info(f"  {i+1}. {df.iloc[i]['QA'][:100]}...")
            
            return df
        
        except Exception as e:
            logger.error(f"Error loading QA dataset: {e}")
            raise
    
    def _detect_query_type_and_weights(self, question: str) -> Tuple[float, float]:
        """Detect query type and return adaptive weights"""
        q_lower = question.lower()
        
        # Strong exact reference
        has_article = bool(re.search(r'\barticle\s+\d+', q_lower))
        has_section = bool(re.search(r'\bsection\s+\d+', q_lower))
        
        if has_article or has_section:
            return 0.7, 0.3
        
        # Numbers present but not explicit article/section
        has_numbers = bool(re.search(r'\d+', q_lower))
        
        # Conceptual keywords
        conceptual_keywords = ['explain', 'what are', 'what is', 'describe', 
                               'how does', 'why', 'meaning', 'definition', 
                               'concept', 'difference', 'tell me about']
        is_conceptual = any(keyword in q_lower for keyword in conceptual_keywords)
        
        if is_conceptual:
            return 0.45, 0.55  # Favor semantic
        elif has_numbers:
            return 0.65, 0.35  # Moderate BM25
        else:
            return self.default_bm25_weight, self.default_vector_weight
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range using min-max normalization"""
        if len(scores) == 0:
            return scores
        
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score - min_score < 1e-10:
            return np.ones_like(scores)
        
        return (scores - min_score) / (max_score - min_score)
    
    # ✅ FIX #3: REPLACED METHOD - HARD Law Type Filtering
    def _filter_by_law_type(self, candidate_indices: List[int], 
                            detected_law_type: Optional[str]) -> List[int]:
        """
        HARD filter candidates by detected law family.
        NO FALLBACKS for high-confidence detections.
        """
        if not self.enable_law_type_filtering or detected_law_type is None:
            return candidate_indices
        
        if detected_law_type not in self.law_type_map:
            logger.warning(f"Unknown law type: {detected_law_type}, skipping filter")
            return candidate_indices
        
        # Get allowed indices for this law family
        allowed_indices = set(self.law_type_map[detected_law_type])
        
        # ✅ HARD FILTER - no 'other' fallback for Constitution/Criminal/Civil
        if detected_law_type in ['constitution', 'criminal', 'civil']:
            # Strict filtering for specific law families
            filtered = [idx for idx in candidate_indices if idx in allowed_indices]
            
            if not filtered:
                logger.warning(
                    f"⚠️  Law type filter for '{detected_law_type}' removed ALL candidates! "
                    f"Original count: {len(candidate_indices)}. "
                    f"This indicates a mismatch between detection and indexing."
                )
                # Return empty - better to fail visibly than return wrong law type
                return []
            
            logger.debug(
                f"Filtered {len(candidate_indices)} → {len(filtered)} "
                f"for law family: {detected_law_type}"
            )
            return filtered
        
        else:
            # For 'other' or unknown, include 'other' as fallback
            allowed_indices.update(self.law_type_map.get('other', []))
            filtered = [idx for idx in candidate_indices if idx in allowed_indices]
            return filtered if filtered else candidate_indices
    
    def bm25_search(self, question: str, top_k: int = 100) -> List[Tuple[int, float]]:
        """Retrieve documents using BM25"""
        tokenized_query = self._tokenize_for_bm25(question)
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        scores = bm25_scores[top_indices]
        normalized_scores = self._normalize_scores(scores)
        
        results = []
        for idx, norm_score in zip(top_indices, normalized_scores):
            if norm_score > 0:
                metadata_idx = self.bm25_doc_mapping[idx]
                results.append((metadata_idx, float(norm_score)))
        
        return results
    
    def vector_search(self, question: str, top_k: int = 100) -> List[Tuple[int, float]]:
        """Retrieve documents using vector similarity"""
        query_emb = self.embedding_model.encode(
            question.strip(),
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype('float32').reshape(1, -1)
        
        distances, indices = self.index.search(query_emb, top_k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if 0 <= idx < len(self.metadata):
                if self.faiss_is_similarity:
                    score = float(dist)
                else:
                    score = 1.0 / (1.0 + float(dist))
                
                results.append((int(idx), score))
        
        if results:
            scores = np.array([s for _, s in results])
            normalized_scores = self._normalize_scores(scores)
            results = [(idx, float(norm_s)) for (idx, _), norm_s in zip(results, normalized_scores)]
        
        return results
    
    def hybrid_search(self, question: str, top_k: int = 10,
                     bm25_weight: float = None, 
                     vector_weight: float = None) -> Tuple[List[Dict], float, float, Optional[str]]:
        """
        FIXED Hybrid retrieval combining BM25 and Vector search.
        
        Fixes Applied:
        - Canonical law family detection
        - Hard law type filtering
        - Zero-vector penalty
        - Exact reference boosting
        """
        # ✅ FIX #2: Detect law type for filtering
        detected_law_type, reference_type = self._detect_law_type_and_reference(question)
        
        # Auto-detect weights if not provided
        if bm25_weight is None or vector_weight is None:
            bm25_weight, vector_weight = self._detect_query_type_and_weights(question)
        
        # Retrieve candidates
        retrieval_k = min(top_k * 10, 100)
        
        bm25_results = self.bm25_search(question, top_k=retrieval_k)
        vector_results = self.vector_search(question, top_k=retrieval_k)
        
        # Create score dictionaries
        bm25_scores = {idx: score for idx, score in bm25_results}
        vector_scores = {idx: score for idx, score in vector_results}
        
        # Get all unique document indices
        all_indices = list(set(bm25_scores.keys()) | set(vector_scores.keys()))
        
        # ✅ FIX #3: HARD filter by law type if detected
        if detected_law_type:
            original_count = len(all_indices)
            all_indices = self._filter_by_law_type(all_indices, detected_law_type)
            if len(all_indices) == 0:
                logger.error(f"Law type filter removed ALL candidates for: {question}")
                # Return empty results rather than wrong law type
                return [], bm25_weight, vector_weight, detected_law_type
            
            logger.debug(f"Law filter: {original_count} → {len(all_indices)} ({detected_law_type})")
        
        # Calculate hybrid scores
        hybrid_scores = {}
        for idx in all_indices:
            bm25_score = bm25_scores.get(idx, 0.0)
            vector_score = vector_scores.get(idx, 0.0)
            
            # Weighted combination
            hybrid_score = (bm25_weight * bm25_score) + (vector_weight * vector_score)
            
            # ✅ FIX #5: Penalize if one component is completely missing
            if bm25_score == 0.0 and vector_score > 0:
                # Vector-only result
                hybrid_score *= 0.7  # Reduce confidence
            elif vector_score == 0.0 and bm25_score > 0:
                # BM25-only result (common for Constitution queries)
                hybrid_score *= 0.5  # Heavy penalty - likely wrong law type
            
            hybrid_scores[idx] = {
                'hybrid_score': hybrid_score,
                'bm25_score': bm25_score,
                'vector_score': vector_score
            }
        
        # ✅ FIX #6: Boost exact article/section matches
        reference_number = self._extract_reference_number(question, reference_type)
        self._boost_exact_reference_match(hybrid_scores, reference_number, reference_type)
        
        # Sort by hybrid score
        sorted_indices = sorted(hybrid_scores.items(), 
                               key=lambda x: x[1]['hybrid_score'], 
                               reverse=True)[:top_k]
        
        # Prepare results
        results = []
        for rank, (idx, scores) in enumerate(sorted_indices, 1):
            doc = self.metadata[idx].copy()
            doc['rank'] = rank
            doc['hybrid_score'] = scores['hybrid_score']
            doc['bm25_score'] = scores['bm25_score']
            doc['vector_score'] = scores['vector_score']
            doc['similarity_score'] = scores['hybrid_score']
            doc['relevance_percentage'] = scores['hybrid_score'] * 100
            results.append(doc)
        
        return results, bm25_weight, vector_weight, detected_law_type
    
    def retrieve_for_question(self, question: str, top_k: int = 10) -> Tuple[List[Dict], str, float, float, Optional[str]]:
        """Retrieve relevant documents for a single question"""
        
        if self.use_hybrid and self.bm25_index is not None:
            results, bm25_w, vector_w, law_type = self.hybrid_search(question, top_k)
            method = "hybrid_fixed"  # ✅ Mark as fixed version
            return results, method, bm25_w, vector_w, law_type
        else:
            vector_results = self.vector_search(question, top_k)
            
            results = []
            for rank, (idx, score) in enumerate(vector_results, 1):
                doc = self.metadata[idx].copy()
                doc['rank'] = rank
                doc['similarity_score'] = score
                doc['relevance_percentage'] = score * 100
                results.append(doc)
            
            method = "vector_only"
            return results, method, 0.0, 1.0, None
    
    def evaluate_single_question(self, question: str, question_id: int, 
                                 top_k: int = 10) -> Tuple[Optional[QueryResult], float]:
        """Evaluate a single question"""
        import time
        
        start_time = time.time()
        
        try:
            retrieved_docs, method, bm25_w, vector_w, law_type = self.retrieve_for_question(question, top_k)
            elapsed_time = (time.time() - start_time) * 1000
            
            result = QueryResult(
                question=question,
                question_id=question_id,
                retrieved_documents=retrieved_docs,
                top_k_used=top_k,
                retrieval_method=method,
                bm25_weight=bm25_w,
                vector_weight=vector_w,
                detected_law_type=law_type
            )
            
            return result, elapsed_time
        
        except Exception as e:
            logger.error(f"Error processing question {question_id}: {e}")
            import traceback
            traceback.print_exc()
            return None, 0
    
    def evaluate_all_questions(self, top_k: int = 10, 
                              save_interval: int = 50) -> Tuple[List[QueryResult], EvaluationMetrics]:
        """Evaluate all questions in the dataset"""
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting FIXED evaluation of {len(self.qa_data)} questions")
        logger.info(f"Retrieving top-{top_k} documents per question")
        logger.info(f"Retrieval mode: {'HYBRID (BM25 + Vector) - FIXED' if self.use_hybrid else 'Vector Only'}")
        logger.info(f"{'='*80}\n")
        
        results = []
        retrieval_times = []
        successful = 0
        failed = 0
        
        top1_scores = []
        top5_scores = []
        top10_scores = []
        law_type_coverage = {}
        
        bm25_weights_used = []
        vector_weights_used = []
        law_type_detections = {'constitution': 0, 'criminal': 0, 'civil': 0, 'none': 0}
        
        for idx, row in tqdm(self.qa_data.iterrows(), total=len(self.qa_data), 
                            desc="Evaluating questions"):
            question = row['QA']
            question_id = row['question_id']
            
            result, elapsed_time = self.evaluate_single_question(question, question_id, top_k)
            
            if result is not None:
                results.append(result)
                retrieval_times.append(elapsed_time)
                successful += 1
                
                bm25_weights_used.append(result.bm25_weight)
                vector_weights_used.append(result.vector_weight)
                
                if result.detected_law_type:
                    law_type_detections[result.detected_law_type] = \
                        law_type_detections.get(result.detected_law_type, 0) + 1
                else:
                    law_type_detections['none'] += 1
                
                if len(result.retrieved_documents) > 0:
                    top1_scores.append(result.retrieved_documents[0]['similarity_score'])
                    
                    if len(result.retrieved_documents) >= 5:
                        top5_avg = np.mean([d['similarity_score'] 
                                          for d in result.retrieved_documents[:5]])
                        top5_scores.append(top5_avg)
                    
                    if len(result.retrieved_documents) >= 10:
                        top10_avg = np.mean([d['similarity_score'] 
                                           for d in result.retrieved_documents[:10]])
                        top10_scores.append(top10_avg)
                    
                    for doc in result.retrieved_documents:
                        law_type = doc.get('law_type', 'Unknown')
                        law_type_coverage[law_type] = law_type_coverage.get(law_type, 0) + 1
            else:
                failed += 1
            
            if (idx + 1) % save_interval == 0:
                self._save_intermediate_results(results, idx + 1)
        
        score_distribution = self._calculate_score_distribution(top1_scores)
        
        hybrid_stats = {
            'avg_bm25_weight': float(np.mean(bm25_weights_used)) if bm25_weights_used else 0.0,
            'avg_vector_weight': float(np.mean(vector_weights_used)) if vector_weights_used else 0.0,
            'bm25_weight_std': float(np.std(bm25_weights_used)) if bm25_weights_used else 0.0,
            'vector_weight_std': float(np.std(vector_weights_used)) if vector_weights_used else 0.0,
            'min_bm25_weight': float(np.min(bm25_weights_used)) if bm25_weights_used else 0.0,
            'max_bm25_weight': float(np.max(bm25_weights_used)) if bm25_weights_used else 0.0
        }
        
        metrics = EvaluationMetrics(
            total_questions=len(self.qa_data),
            successful_retrievals=successful,
            failed_retrievals=failed,
            avg_top1_score=np.mean(top1_scores) if top1_scores else 0.0,
            avg_top5_score=np.mean(top5_scores) if top5_scores else 0.0,
            avg_top10_score=np.mean(top10_scores) if top10_scores else 0.0,
            avg_retrieval_time_ms=np.mean(retrieval_times) if retrieval_times else 0.0,
            coverage_by_law_type=law_type_coverage,
            score_distribution=score_distribution,
            hybrid_stats=hybrid_stats,
            law_type_detection_stats=law_type_detections,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"\n{'='*80}")
        logger.info("FIXED EVALUATION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total Questions: {metrics.total_questions}")
        logger.info(f"Successful: {metrics.successful_retrievals}")
        logger.info(f"Failed: {metrics.failed_retrievals}")
        logger.info(f"Average Top-1 Score: {metrics.avg_top1_score:.4f}")
        logger.info(f"Average Retrieval Time: {metrics.avg_retrieval_time_ms:.2f} ms")
        
        if self.use_hybrid:
            logger.info(f"\nHybrid Retrieval Statistics:")
            logger.info(f"  Avg BM25 Weight: {hybrid_stats['avg_bm25_weight']:.3f} ± {hybrid_stats['bm25_weight_std']:.3f}")
            logger.info(f"  Law Type Detection:")
            for law_type, count in law_type_detections.items():
                logger.info(f"    {law_type}: {count} questions")
        
        logger.info(f"{'='*80}\n")
        
        return results, metrics
    
    def _calculate_score_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Calculate distribution of similarity scores"""
        distribution = {
            "0.9-1.0 (Excellent)": 0,
            "0.8-0.9 (Very Good)": 0,
            "0.7-0.8 (Good)": 0,
            "0.6-0.7 (Fair)": 0,
            "0.5-0.6 (Moderate)": 0,
            "0.0-0.5 (Poor)": 0
        }
        
        for score in scores:
            if score >= 0.9:
                distribution["0.9-1.0 (Excellent)"] += 1
            elif score >= 0.8:
                distribution["0.8-0.9 (Very Good)"] += 1
            elif score >= 0.7:
                distribution["0.7-0.8 (Good)"] += 1
            elif score >= 0.6:
                distribution["0.6-0.7 (Fair)"] += 1
            elif score >= 0.5:
                distribution["0.5-0.6 (Moderate)"] += 1
            else:
                distribution["0.0-0.5 (Poor)"] += 1
        
        return distribution
    
    def _save_intermediate_results(self, results: List[QueryResult], count: int):
        """Save intermediate results"""
        output_file = os.path.join(self.output_dir, f"intermediate_results_FIXED_{count}.json")
        results_dict = [r.to_dict() for r in results]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved intermediate results: {count} questions processed")
    
    def save_results(self, results: List[QueryResult], metrics: EvaluationMetrics):
        """Save final results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results_file = os.path.join(self.output_dir, 
                                   f"qa_evaluation_results_FIXED_{timestamp}.json")
        results_dict = [r.to_dict() for r in results]
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Saved detailed results to: {results_file}")
        
        metrics_file = os.path.join(self.output_dir, 
                                    f"qa_evaluation_metrics_FIXED_{timestamp}.json")
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(metrics), f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Saved metrics to: {metrics_file}")
        
        self._generate_summary_report(results, metrics, timestamp)
        
        return results_file, metrics_file
    
    def _generate_summary_report(self, results: List[QueryResult], 
                                 metrics: EvaluationMetrics, timestamp: str):
        """Generate summary report"""
        report_file = os.path.join(self.output_dir, 
                                   f"evaluation_summary_FIXED_{timestamp}.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("QA EVALUATION SUMMARY - FIXED HYBRID RETRIEVAL\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Evaluation Timestamp: {metrics.timestamp}\n")
            f.write(f"Total Questions: {metrics.total_questions}\n")
            f.write(f"Successful: {metrics.successful_retrievals}\n")
            f.write(f"Success Rate: {(metrics.successful_retrievals/metrics.total_questions*100):.2f}%\n\n")
            
            f.write("FIXES APPLIED:\n")
            f.write("✅ Canonical law family tagging\n")
            f.write("✅ Aligned law type detection\n")
            f.write("✅ Hard law type filtering\n")
            f.write("✅ Zero-vector score penalty\n")
            f.write("✅ Exact reference boosting\n\n")
            
            f.write("-"*80 + "\n")
            f.write("PERFORMANCE METRICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Average Top-1 Score: {metrics.avg_top1_score:.4f}\n")
            f.write(f"Average Top-5 Score: {metrics.avg_top5_score:.4f}\n")
            f.write(f"Average Retrieval Time: {metrics.avg_retrieval_time_ms:.2f} ms\n\n")
            
            f.write("-"*80 + "\n")
            f.write("LAW TYPE DETECTION\n")
            f.write("-"*80 + "\n")
            for law_type, count in metrics.law_type_detection_stats.items():
                percentage = (count / metrics.total_questions * 100) if metrics.total_questions > 0 else 0
                f.write(f"{law_type}: {count} ({percentage:.1f}%)\n")
            f.write("\n")
            
            f.write("-"*80 + "\n")
            f.write("SCORE DISTRIBUTION\n")
            f.write("-"*80 + "\n")
            for score_range, count in metrics.score_distribution.items():
                percentage = (count / metrics.total_questions * 100) if metrics.total_questions > 0 else 0
                f.write(f"{score_range}: {count} ({percentage:.1f}%)\n")
            f.write("\n")
            
            f.write("-"*80 + "\n")
            f.write("SAMPLE RESULTS\n")
            f.write("-"*80 + "\n")
            for i, result in enumerate(results[:5], 1):
                f.write(f"\nQuestion {i}: {result.question}\n")
                f.write(f"Detected Law Type: {result.detected_law_type or 'None'}\n")
                
                if result.retrieved_documents:
                    top_doc = result.retrieved_documents[0]
                    f.write(f"Top Result Law Type: {top_doc.get('law_type', 'N/A')}\n")
                    f.write(f"Section: {top_doc.get('source_file', 'N/A')}\n")
                    f.write(f"Vector Score: {top_doc.get('vector_score', 0):.4f}\n")
                    f.write(f"BM25 Score: {top_doc.get('bm25_score', 0):.4f}\n")
                    f.write(f"Hybrid Score: {top_doc.get('hybrid_score', 0):.4f}\n")
                f.write("\n")
            
            f.write("="*80 + "\n")
        
        logger.info(f"✅ Saved summary report to: {report_file}")
    
    def generate_analysis_csv(self, results: List[QueryResult], 
                             metrics: EvaluationMetrics):
        """Generate CSV for analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = os.path.join(self.output_dir, 
                               f"qa_evaluation_analysis_FIXED_{timestamp}.csv")
        
        rows = []
        for result in results:
            row = {
                'question_id': result.question_id,
                'question': result.question,
                'detected_law_type': result.detected_law_type or 'None',
                'retrieval_method': result.retrieval_method,
                'bm25_weight': result.bm25_weight,
                'vector_weight': result.vector_weight,
                'num_retrieved': len(result.retrieved_documents),
            }
            
            if result.retrieved_documents:
                top_doc = result.retrieved_documents[0]
                row['top1_score'] = top_doc.get('hybrid_score', 
                                               top_doc.get('similarity_score', 0))
                row['top1_bm25_score'] = top_doc.get('bm25_score', 0)
                row['top1_vector_score'] = top_doc.get('vector_score', 0)
                row['top1_law_type'] = top_doc.get('law_type', 'N/A')
                row['top1_law_family'] = top_doc.get('law_family', 'N/A')  # ✅ NEW
                row['top1_section'] = top_doc.get('section_number', 
                                                  top_doc.get('article_number', 'N/A'))
                row['top1_source_file'] = top_doc.get('source_file', 'N/A')
            else:
                row['top1_score'] = 0
                row['top1_bm25_score'] = 0
                row['top1_vector_score'] = 0
                row['top1_law_type'] = 'N/A'
                row['top1_law_family'] = 'N/A'
                row['top1_section'] = 'N/A'
                row['top1_source_file'] = 'N/A'
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        logger.info(f"✅ Saved analysis CSV to: {csv_file}")
        return csv_file


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="QA Evaluation Pipeline - FIXED Hybrid BM25 + Vector Retrieval"
    )
    parser.add_argument("--qa-csv", required=True, help="Path to QA CSV file")
    parser.add_argument("--index-path", type=str, default="./faiss_index")
    parser.add_argument("--metadata-path", type=str, default="./faiss_metadata.pkl")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results_FIXED")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-hybrid", action="store_true")
    parser.add_argument("--no-filtering", action="store_true")
    parser.add_argument("--bm25-weight", type=float, default=0.6)
    parser.add_argument("--vector-weight", type=float, default=0.4)
    
    args = parser.parse_args()
    
    total = args.bm25_weight + args.vector_weight
    if abs(total - 1.0) > 0.001:
        logger.warning(f"Weights don't sum to 1.0, normalizing...")
        args.bm25_weight /= total
        args.vector_weight /= total
    
    print("\n" + "="*80)
    print("QA EVALUATION PIPELINE - FIXED VERSION")
    print("="*80 + "\n")
    
    pipeline = QAEvaluationPipeline(
        qa_csv_path=args.qa_csv,
        index_path=args.index_path,
        metadata_path=args.metadata_path,
        output_dir=args.output_dir,
        gpu_id=args.gpu_id,
        use_cpu=args.cpu,
        use_hybrid=not args.no_hybrid,
        default_bm25_weight=args.bm25_weight,
        default_vector_weight=args.vector_weight,
        enable_law_type_filtering=not args.no_filtering
    )
    
    results, metrics = pipeline.evaluate_all_questions(top_k=args.top_k)
    
    print("\nSaving results...")
    results_file, metrics_file = pipeline.save_results(results, metrics)
    
    print("\nGenerating analysis CSV...")
    csv_file = pipeline.generate_analysis_csv(results, metrics)
    
    print("\n" + "="*80)
    print("FIXED EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  1. Results: {results_file}")
    print(f"  2. Metrics: {metrics_file}")
    print(f"  3. CSV: {csv_file}")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()