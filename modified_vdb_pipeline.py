import torch
import sys
import os
import json
import logging
import re
import pickle
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

# CRITICAL: Use ONLY GPU 0 and GPU 7
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class LegalDocument:
    """Data structure for legal documents"""
    section_number: str
    title: str
    content: str
    chapter: str
    keywords: List[str]
    law_type: str
    doc_id: int
    source_file: str
    embedding: Optional[List[float]] = None


class OptimizedDataIngestionPipeline:
    """Optimized pipeline for processing text files from folder"""
    
    def __init__(self, 
                 data_folder: str = "data/stage1_unsupervised/acts_raw",
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 index_path: str = "./faiss_index",
                 metadata_path: str = "./faiss_metadata.pkl"):
        
        self.data_folder = data_folder
        self.embedding_model_name = embedding_model_name
        self.index_path = index_path
        self.metadata_path = metadata_path
        
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.documents_metadata = []
    
    def _extract_keywords(self, text: str, title: str = "", chapter: str = "") -> List[str]:
        """Extract keywords from legal text"""
        legal_keywords = [
            'punishment', 'fine', 'imprisonment', 'offense', 'crime', 'murder', 'theft',
            'constitution', 'article', 'fundamental', 'rights', 'duties', 'parliament',
            'court', 'judge', 'bail', 'warrant', 'evidence', 'witness', 'contract',
            'property', 'criminal', 'civil', 'procedure', 'appeal', 'jurisdiction',
            'section', 'chapter', 'act', 'law', 'provision', 'clause', 'penalty',
            'accused', 'victim', 'trial', 'prosecution', 'defense', 'verdict',
            'marriage', 'divorce', 'adoption', 'succession', 'inheritance', 'guardianship',
            'arbitration', 'mediation', 'healthcare', 'welfare', 'maintenance', 'prohibition'
        ]
        
        combined_text = f"{chapter} {title} {text}".lower()
        keywords = []
        
        for keyword in legal_keywords:
            if keyword in combined_text:
                keywords.append(keyword)
        
        section_patterns = [
            r'section\s+\d+[a-z]?', r'article\s+\d+', r'chapter\s+[ivx\d]+',
            r'act,?\s+\d{4}', r'rule\s+\d+', r'clause\s+\d+',
            r'sub-section\s+\(\d+\)', r'paragraph\s+\d+'
        ]
        
        for pattern in section_patterns:
            matches = re.findall(pattern, combined_text)
            keywords.extend(matches)
        
        return list(set(keywords))
    
    def _determine_law_type(self, filename: str, content: str) -> str:
        """Determine the type of law from filename and content"""
        filename_lower = filename.lower()
        content_lower = content.lower()
        
        # Extract from filename first
        if 'marriage' in filename_lower:
            if 'hindu' in filename_lower:
                return 'Hindu Marriage Act'
            elif 'christian' in filename_lower:
                return 'Christian Marriage Act'
            elif 'foreign' in filename_lower:
                return 'Foreign Marriage Act'
            elif 'prohibition' in filename_lower:
                return 'Child Marriage Prohibition Act'
            else:
                return 'Marriage Act'
        elif 'divorce' in filename_lower:
            return 'Divorce Act'
        elif 'succession' in filename_lower:
            if 'hindu' in filename_lower:
                return 'Hindu Succession Act'
            elif 'indian' in filename_lower:
                return 'Indian Succession Act'
            else:
                return 'Succession Act'
        elif 'adoption' in filename_lower:
            return 'Hindu Adoption Act'
        elif 'guardians' in filename_lower or 'wards' in filename_lower:
            return 'Guardians and Wards Act'
        elif 'minority' in filename_lower:
            return 'Hindu Minority Act'
        elif 'disposition' in filename_lower:
            return 'Hindu Disposition of Property Act'
        elif 'arbitration' in filename_lower or 'mediation' in filename_lower:
            return 'Arbitration and Mediation Act'
        elif 'evidence' in filename_lower:
            return 'Indian Evidence Act'
        elif 'bar_councils' in filename_lower or 'bar councils' in filename_lower:
            return 'Indian Bar Councils Act'
        elif 'juvenile' in filename_lower:
            return 'Juvenile Justice Act'
        elif 'legal_service' in filename_lower or 'legal service' in filename_lower:
            return 'Legal Service Authorities Act'
        elif 'family_courts' in filename_lower or 'family courts' in filename_lower:
            return 'Family Courts Act'
        elif 'domestic_violence' in filename_lower or 'domestic violence' in filename_lower:
            return 'Domestic Violence Act'
        elif 'dowry' in filename_lower:
            return 'Dowry Prohibition Act'
        elif 'maintenance' in filename_lower or 'welfare' in filename_lower:
            return 'Maintenance and Welfare Act'
        elif 'healthcare' in filename_lower:
            return 'Healthcare Act'
        elif 'bns' in filename_lower or 'bnss' in filename_lower:
            return 'Bharatiya Nyaya Sanhita'
        elif 'bsa' in filename_lower:
            return 'Bharatiya Sakshya Adhiniyam'
        elif 'huf' in filename_lower:
            return 'Hindu Undivided Family Act'
        else:
            return 'Other Legal Act'
    
    def _parse_text_file(self, filepath: str, filename: str) -> List[LegalDocument]:
        """Parse a text file and extract sections"""
        documents = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to split by common section markers
            section_patterns = [
                r'\n(?:Section|SECTION)\s+\d+[A-Za-z]?\.?\s*[-–—]?\s*',
                r'\n(?:Article|ARTICLE)\s+\d+\.?\s*[-–—]?\s*',
                r'\n\d+\.\s+',
                r'\n(?:Chapter|CHAPTER)\s+[IVXLCDM\d]+\s*[-–—]?\s*'
            ]
            
            sections = []
            for pattern in section_patterns:
                parts = re.split(pattern, content)
                if len(parts) > 1:
                    sections = parts
                    break
            
            # If no sections found, treat entire file as one document
            if not sections or len(sections) <= 1:
                sections = [content]
            
            law_type = self._determine_law_type(filename, content)
            
            for idx, section_content in enumerate(sections):
                section_content = section_content.strip()
                
                if len(section_content) < 50:  # Skip very short sections
                    continue
                
                # Extract section number and title
                lines = section_content.split('\n', 2)
                section_number = f"{filename} - Part {idx + 1}"
                title = filename.replace('_', ' ').replace('.txt', '')
                chapter = ""
                
                # Try to extract better section info from first line
                if lines:
                    first_line = lines[0].strip()
                    # Look for section/article numbers
                    section_match = re.match(r'((?:Section|Article|Chapter)\s+\d+[A-Za-z]?\.?)\s*[-–—]?\s*(.*)', 
                                            first_line, re.IGNORECASE)
                    if section_match:
                        section_number = section_match.group(1)
                        if section_match.group(2):
                            title = section_match.group(2)[:200]
                
                keywords = self._extract_keywords(section_content, title, chapter)
                
                doc = LegalDocument(
                    section_number=section_number,
                    title=title,
                    content=section_content[:4000],  # Limit content length
                    chapter=chapter,
                    keywords=keywords,
                    law_type=law_type,
                    doc_id=len(documents),
                    source_file=filename
                )
                
                documents.append(doc)
        
        except Exception as e:
            logger.warning(f"Error parsing file {filename}: {e}")
        
        return documents
    
    def load_dataset(self) -> List[LegalDocument]:
        """Load all text files from the data folder"""
        logger.info(f"Loading text files from: {self.data_folder}")
        
        if not os.path.exists(self.data_folder):
            raise ValueError(f"Data folder not found: {self.data_folder}")
        
        all_documents = []
        files_processed = 0
        
        # Get all .txt files
        txt_files = [f for f in os.listdir(self.data_folder) if f.endswith('.txt')]
        
        logger.info(f"Found {len(txt_files)} text files")
        
        for filename in sorted(txt_files):
            filepath = os.path.join(self.data_folder, filename)
            logger.info(f"Processing: {filename}")
            
            documents = self._parse_text_file(filepath, filename)
            all_documents.extend(documents)
            files_processed += 1
            
            logger.info(f"  → Extracted {len(documents)} sections from {filename}")
        
        logger.info(f"✅ Processed {files_processed} files, extracted {len(all_documents)} documents")
        
        # Show statistics
        law_types = {}
        for doc in all_documents:
            law_types[doc.law_type] = law_types.get(doc.law_type, 0) + 1
        
        logger.info("Law type distribution:")
        for lt, count in sorted(law_types.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {lt}: {count} documents")
        
        if len(all_documents) == 0:
            raise ValueError("No valid documents extracted!")
        
        return all_documents
    
    def generate_embeddings(self, documents: List[LegalDocument], batch_size: int = 32) -> List[LegalDocument]:
        """Generate embeddings with progress tracking"""
        logger.info(f"Generating embeddings for {len(documents)} documents...")
        
        texts = []
        for doc in documents:
            text = f"{doc.law_type} | {doc.section_number} | {doc.title}: {doc.content[:500]}"
            texts.append(text)
        
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_emb = self.embedding_model.encode(
                batch, 
                convert_to_numpy=True, 
                normalize_embeddings=True,
                show_progress_bar=False
            )
            all_embeddings.extend(batch_emb)
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Embedding progress: {i+len(batch)}/{len(texts)} "
                          f"({(i+len(batch))/len(texts)*100:.1f}%)")
        
        for doc, emb in zip(documents, all_embeddings):
            doc.embedding = emb.tolist()
        
        logger.info(f"✅ Generated {len(all_embeddings)} embeddings")
        return documents
    
    def build_faiss_index(self, documents: List[LegalDocument]):
        """Build FAISS index"""
        logger.info(f"Building FAISS index for {len(documents)} documents...")
        
        embeddings = np.array([doc.embedding for doc in documents]).astype('float32')
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        for doc in documents:
            doc_dict = asdict(doc)
            doc_dict.pop('embedding', None)
            self.documents_metadata.append(doc_dict)
        
        logger.info(f"✅ Index built: {self.index.ntotal} vectors")
    
    def save_index(self):
        """Save index and metadata"""
        os.makedirs(os.path.dirname(self.index_path) or '.', exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.documents_metadata, f)
        
        logger.info(f"✅ Saved index to: {self.index_path}")
        logger.info(f"✅ Saved metadata to: {self.metadata_path}")
    
    def run_ingestion(self):
        """Run complete ingestion pipeline"""
        try:
            documents = self.load_dataset()
            documents = self.generate_embeddings(documents)
            self.build_faiss_index(documents)
            self.save_index()
            logger.info("🎉 Ingestion pipeline completed successfully!")
            return True
        except Exception as e:
            logger.error(f"❌ Ingestion failed: {e}")
            import traceback
            traceback.print_exc()
            return False


class CitationRetrievalPipeline:
    """Retrieval pipeline that returns only citations from vector database"""
    
    def __init__(self,
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 index_path: str = "./faiss_index",
                 metadata_path: str = "./faiss_metadata.pkl"):
        
        logger.info("Loading retrieval pipeline...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = faiss.read_index(index_path)
        
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        logger.info(f"✅ Loaded: {self.index.ntotal} vectors, {len(self.metadata)} documents")
    
    def retrieve(self, query: str, limit: int = 10) -> List[Dict]:
        """Retrieve relevant documents with citations"""
        query_emb = self.embedding_model.encode(
            query, convert_to_numpy=True, normalize_embeddings=True
        ).astype('float32').reshape(1, -1)
        
        distances, indices = self.index.search(query_emb, limit)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if 0 <= idx < len(self.metadata):
                doc = self.metadata[idx].copy()
                doc['similarity_score'] = float(dist)
                doc['relevance_percentage'] = float(dist * 100)
                results.append(doc)
        
        return results
    
    def get_citations(self, query: str, limit: int = 10) -> Dict:
        """Get citations for a query"""
        docs = self.retrieve(query, limit)
        
        if not docs:
            return {
                "query": query,
                "total_citations": 0,
                "citations": []
            }
        
        citations = []
        for i, doc in enumerate(docs, 1):
            citation = {
                "rank": i,
                "section_number": doc.get('section_number', 'N/A'),
                "title": doc.get('title', 'N/A'),
                "chapter": doc.get('chapter', 'N/A') if doc.get('chapter') else 'N/A',
                "law_type": doc.get('law_type', 'N/A'),
                "source_file": doc.get('source_file', 'N/A'),
                "keywords": doc.get('keywords', [])[:10],
                "content_preview": doc.get('content', '')[:500] + "...",
                "full_content": doc.get('content', ''),
                "similarity_score": doc.get('similarity_score', 0.0),
                "relevance_percentage": f"{doc.get('relevance_percentage', 0.0):.2f}%"
            }
            citations.append(citation)
        
        return {
            "query": query,
            "total_citations": len(citations),
            "citations": citations
        }
    
    def query(self, user_query: str, limit: int = 10, show_full_content: bool = False) -> Dict:
        """Query and return citations"""
        result = self.get_citations(user_query, limit)
        
        # Remove full_content if not requested
        if not show_full_content:
            for citation in result['citations']:
                citation.pop('full_content', None)
        
        return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Citation-based RAG Pipeline for Indian Laws")
    parser.add_argument("--action", required=True, 
                       choices=["ingest", "test", "query"],
                       help="Action to perform")
    parser.add_argument("--data-folder", type=str, 
                       default="data/stage1_unsupervised/acts_raw",
                       help="Folder containing text files")
    parser.add_argument("--query-text", type=str, default="",
                       help="Query text for --action query")
    parser.add_argument("--limit", type=int, default=10,
                       help="Number of citations to retrieve")
    parser.add_argument("--show-full-content", action="store_true",
                       help="Show full content in citations")
    
    args = parser.parse_args()
    
    if args.action == "ingest":
        print("\n📥 Running data ingestion pipeline...")
        print("="*80)
        pipeline = OptimizedDataIngestionPipeline(data_folder=args.data_folder)
        success = pipeline.run_ingestion()
        
        if success:
            print("\n" + "="*80)
            print("✅ INGESTION COMPLETE!")
            print("="*80)
            print("\nNext steps:")
            print("  1. Test: python modified_rag_pipeline.py --action test")
            print("  2. Query: python modified_rag_pipeline.py --action query --query-text 'your question'")
        else:
            print("\n❌ Ingestion failed. Check the logs above.")
    
    elif args.action == "test":
        print("\n🧪 Running test queries...")
        print("="*80)
        
        retrieval = CitationRetrievalPipeline()
        
        test_queries = [
            "What are the provisions for marriage under Hindu law?",
            "Explain divorce procedures",
            "What are the rules for adoption?",
            "Guardianship and custody laws",
            "Succession and inheritance rights"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*80}")
            print(f"TEST QUERY {i}/{len(test_queries)}: {query}")
            print(f"{'='*80}")
            
            result = retrieval.query(query, limit=5, show_full_content=False)
            
            print(f"\n📊 Total Citations: {result['total_citations']}")
            
            print("\n📚 CITATIONS:")
            for citation in result['citations']:
                print(f"\n  [{citation['rank']}] {citation['law_type']}")
                print(f"      Section: {citation['section_number']}")
                print(f"      Title: {citation['title']}")
                print(f"      Source File: {citation['source_file']}")
                print(f"      Relevance: {citation['relevance_percentage']}")
                print(f"      Keywords: {', '.join(citation['keywords'][:5])}")
                print(f"      Preview: {citation['content_preview'][:200]}...")
            
            print(f"\n{'='*80}\n")
            
            if i < len(test_queries):
                import time
                time.sleep(1)
        
        print("\n✅ Testing complete!")
    
    elif args.action == "query":
        if not args.query_text:
            print("❌ Error: --query-text is required for query action")
            print("Example: python modified_rag_pipeline.py --action query --query-text 'What is Hindu Marriage Act?'")
            return
        
        print(f"\n🔍 Query: {args.query_text}")
        print("="*80)
        
        retrieval = CitationRetrievalPipeline()
        result = retrieval.query(args.query_text, limit=args.limit, 
                                show_full_content=args.show_full_content)
        
        print(f"\n📊 Total Citations Found: {result['total_citations']}")
        
        print("\n📚 CITATIONS:")
        print("="*80)
        
        for citation in result['citations']:
            print(f"\n[{citation['rank']}] {citation['law_type']}")
            print(f"    Section: {citation['section_number']}")
            print(f"    Title: {citation['title']}")
            if citation['chapter'] != 'N/A':
                print(f"    Chapter: {citation['chapter']}")
            print(f"    Source File: {citation['source_file']}")
            print(f"    Relevance: {citation['relevance_percentage']}")
            print(f"    Keywords: {', '.join(citation['keywords'])}")
            print(f"\n    Preview:")
            print(f"    {citation['content_preview']}")
            
            if args.show_full_content:
                print(f"\n    Full Content:")
                print(f"    {citation.get('full_content', 'N/A')}")
            
            print(f"\n    {'-'*76}")
        
        print(f"\n{'='*80}\n")
        
        # Save to JSON file
        output_file = f"citations_{args.query_text[:30].replace(' ', '_')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"💾 Citations saved to: {output_file}")


if __name__ == "__main__":
    main()