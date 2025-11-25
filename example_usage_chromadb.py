"""Example usage script for loading existing ChromaDB and testing retrieval.

This script demonstrates:
1. Loading existing ChromaDB from chromadb folder using ChromaConnection
2. Using SentenceTransformer (HuggingFace) for embeddings
3. Retrieving chunks and testing retrieval quality

Usage:
    # Basic usage - run the script as-is
    python example_usage_chromadb.py
    
    # The script will:
    # 1. Load ChromaDB from ./chromadb directory
    # 2. Use the first available collection (or specify COLLECTION_NAME)
    # 3. Load SentenceTransformer model "all-MiniLM-L6-v2"
    # 4. Sample 10 chunks from the collection
    # 5. Generate QA pairs and test retrieval
    # 6. Save results to retrieval_results.csv

Configuration (edit in main() function):
    CHROMADB_DIR = "./chromadb"  # Path to your ChromaDB directory
    COLLECTION_NAME = None  # None = use first available, or specify name
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # HuggingFace model name
    NUM_CHUNKS = 10  # Number of chunks to retrieve and test
"""

import json
import random
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️  ChromaDB not installed. Install with: pip install chromadb")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️  pandas not installed. Install with: pip install pandas")

from evalvd.vdb_integrations.chroma_con import ChromaConnection
from sentence_transformers import SentenceTransformer
from evalvd.eval.metrics import calculate_retrieval_metrics
from evalvd.generation.qa_generator import QAGenerator
from evalvd.utils.versioning import create_version, save_to_version


def load_existing_chromadb(
    persist_directory: str,
    collection_name: Optional[str] = None,
    embedding_model_name: str = "all-MiniLM-L6-v2"
) -> ChromaConnection:
    """
    Load existing ChromaDB from persistent directory.
    
    Args:
        persist_directory: Directory containing ChromaDB files
        collection_name: Name of collection to load (if None, uses first available)
        embedding_model_name: Model name for SentenceTransformer (HuggingFace)
        
    Returns:
        ChromaConnection instance
    """
    if not CHROMADB_AVAILABLE:
        raise ImportError("ChromaDB is not installed")
    
    print(f"Loading existing ChromaDB from {persist_directory}...")
    
    # Initialize embedding model
    embedding_model = SentenceTransformer(embedding_model_name)
    print(f"   Using SentenceTransformer with model: {embedding_model_name}")
    
    # First, connect to ChromaDB to list available collections
    import chromadb
    client = chromadb.PersistentClient(path=persist_directory)
    collections = client.list_collections()
    
    if not collections:
        raise ValueError(f"No collections found in {persist_directory}")
    
    print(f"   Available collections: {[c.name for c in collections]}")
    
    # Determine which collection to use
    if collection_name is None:
        collection_name = collections[0].name
        print(f"   Using first available collection: {collection_name}")
    else:
        # Check if collection exists
        collection_names = [c.name for c in collections]
        if collection_name not in collection_names:
            raise ValueError(f"Collection '{collection_name}' not found. Available: {collection_names}")
        print(f"   Loading collection: {collection_name}")
    
    # Create ChromaConnection with the specified collection
    connection = ChromaConnection(
        path=persist_directory,
        collection_name=collection_name,  # Use the actual collection name, not "random"
        embedding_model=embedding_model
    )
    
    print(f"✅ Connected to persistent ChromaDB")
    print(f"   Collection: {connection.collection_name}")
    
    # Get collection count
    try:
        count = connection.collection.count()
        print(f"   Collection count: {count} documents")
    except Exception:
        print(f"   Collection count: (unavailable)")
    
    return connection


def get_chunks_from_collection(connection: ChromaConnection, num_chunks: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve random chunks from ChromaDB collection.
    
    Args:
        connection: ChromaConnection instance
        num_chunks: Number of chunks to retrieve (default: 5)
        
    Returns:
        List of chunk dictionaries with text, metadata, and id
    """
    print(f"\nRetrieving {num_chunks} chunks from collection...")
    
    # Use sample_chunks method
    chunks = connection.sample_chunks(num_chunks)
    
    print(f"✅ Retrieved {len(chunks)} chunks")
    return chunks


def create_retrieval_dataframe(
    connection: ChromaConnection,
    chunks: List[Dict[str, Any]],
    top_k: int = 5,
    generate_qa: bool = True,
    device: Optional[str] = None,
    batch_size: int = 8
) -> pd.DataFrame:
    """
    Create pandas DataFrame with chunks, questions, answers, and retrieved documents.
    
    Args:
        connection: ChromaConnection instance
        chunks: List of chunks to generate questions from
        top_k: Number of top results to retrieve per query
        generate_qa: If True, use QAGenerator; if False, create simple queries
        device: Device for QAGenerator (None for auto-detect)
        batch_size: Batch size for QAGenerator
        
    Returns:
        pandas DataFrame with columns: chunk, chunk_id, question, answer, retrieved_docs, retrieved_ids, num_retrieved, chunk_position
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required. Install with: pip install pandas")
    
    print(f"\n{'='*60}")
    print("Creating Retrieval DataFrame")
    print(f"{'='*60}")
    
    # Create query function using connection
    def query_fn(question: str, top_k: int) -> List[Dict[str, Any]]:
        """Query ChromaDB collection."""
        results = connection.query(question, top_k=top_k)
        return results
    
    # Generate questions/answers for each chunk
    rows = []
    qa_pairs = []  # Initialize to empty list
    
    if generate_qa:
        print(f"   Generating questions/answers for {len(chunks)} chunks...")
        try:
            qa_generator = QAGenerator(device=device, batch_size=batch_size)
            # Process in batches to show progress
            # Initialize qa_pairs list with None values to maintain chunk index alignment
            qa_pairs = [None] * len(chunks)
            batch_size_gen = batch_size if batch_size else 8
            
            # Extract passages for batch processing
            passages = [chunk.get("text", "") for chunk in chunks]
            
            with tqdm(total=len(passages), desc="   Generating QA pairs", unit="chunk") as pbar:
                # Process in batches
                for i in range(0, len(passages), batch_size_gen):
                    batch_passages = passages[i:i + batch_size_gen]
                    batch_chunks = chunks[i:i + batch_size_gen]
                    
                    # Generate QA for this batch
                    batch_qa = qa_generator.generate_qa_batch(batch_passages, max_retries=1)
                    
                    # Match batch QA pairs with chunks by index
                    # Note: batch_qa may have fewer items if some generations failed
                    for j, chunk in enumerate(batch_chunks):
                        chunk_idx = i + j
                        if j < len(batch_qa) and batch_qa[j]:
                            # Add metadata if available
                            if isinstance(chunk, dict) and "metadata" in chunk:
                                batch_qa[j]["metadata"] = chunk["metadata"]
                            qa_pairs[chunk_idx] = batch_qa[j]
                    
                    pbar.update(len(batch_passages))
                    generated_count = sum(1 for qa in qa_pairs if qa is not None)
                    pbar.set_postfix({"generated": generated_count})
            
            # Filter out None values for cleaner list, but keep index mapping
            generated_count = sum(1 for qa in qa_pairs if qa is not None)
            print(f"   ✅ Generated {generated_count}/{len(chunks)} QA pairs")
        except Exception as e:
            print(f"   ⚠️  QAGenerator failed: {e}")
            print("   Falling back to simple query generation...")
            generate_qa = False
            qa_pairs = [None] * len(chunks)  # Initialize with None values to maintain alignment
    
    # Process chunks with retrieval, showing progress
    print(f"   Processing {len(chunks)} chunks with retrieval...")
    for i, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="   Processing chunks", unit="chunk"):
        chunk_text = chunk.get("text", "")
        chunk_id = chunk.get("id", "")
        
        # Get question and answer
        if generate_qa and qa_pairs and i < len(qa_pairs) and qa_pairs[i] is not None:
            question = qa_pairs[i].get("question", "")
            answer = qa_pairs[i].get("answer", "")
        else:
            # Create simple question from chunk text
            words = chunk_text.split()[:10]
            question = " ".join(words) + "?" if words else "What is this about?"
            answer = chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
        
        # Query with the question
        retrieved = query_fn(question, top_k=top_k)
        
        # Extract retrieved docs and ids
        retrieved_docs = [r["text"] for r in retrieved]
        retrieved_ids = [r.get("id", "") for r in retrieved]
        
        # Find the position of the original chunk in retrieved results
        chunk_position = None
        for pos, retrieved_id in enumerate(retrieved_ids):
            if retrieved_id == chunk_id:
                chunk_position = pos + 1  # 1-indexed position
                break
        
        # If not found by ID, try to match by text content
        if chunk_position is None:
            for pos, retrieved_doc in enumerate(retrieved_docs):
                if retrieved_doc == chunk_text or retrieved_doc.strip() == chunk_text.strip():
                    chunk_position = pos + 1  # 1-indexed position
                    break
        
        rows.append({
            "chunk": chunk_text,
            "chunk_id": chunk_id,
            "question": question,
            "answer": answer,
            "retrieved_docs": retrieved_docs,
            "retrieved_ids": retrieved_ids,
            "num_retrieved": len(retrieved_docs),
            "chunk_position": chunk_position  # Position of original chunk in retrieved results (1-indexed, None if not found)
        })
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    print(f"\n   ✅ Created DataFrame with {len(df)} rows")
    print(f"   Columns: {', '.join(df.columns)}")
    
    return df


def test_retrieval(
    connection: ChromaConnection,
    num_test_queries: int = 10,
    top_k: int = 5,
    chunks_for_testing: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Test retrieval quality using the evaluation tool.
    
    Args:
        connection: ChromaConnection instance
        num_test_queries: Number of test queries to run
        top_k: Number of top results to retrieve per query
        chunks_for_testing: Optional list of chunks to use as ground truth
        
    Returns:
        Dictionary with retrieval test results
    """
    print(f"\n{'='*60}")
    print("Testing Retrieval Quality")
    print(f"{'='*60}")
    
    # Use connection directly
    vdb = connection
    
    # Generate test queries from chunks
    if chunks_for_testing is None:
        print("   Sampling chunks to create test queries...")
        chunks_for_testing = vdb.sample_chunks(num_test_queries)
    
    if len(chunks_for_testing) == 0:
        print("⚠️  No chunks available for testing")
        return {}
    
    # Create test QA pairs (using chunk text as answer context)
    test_queries = []
    for i, chunk in enumerate(chunks_for_testing[:num_test_queries]):
        # Create a simple query based on chunk text
        text = chunk["text"]
        # Take first sentence or first 10 words as query
        words = text.split()[:10]
        query = " ".join(words) + "?"
        test_queries.append({
            "question": query,
            "answer": text[:100] + "...",  # First 100 chars as answer
            "passage": text  # Full passage as relevant chunk
        })
    
    print(f"   Created {len(test_queries)} test queries")
    
    # Test retrieval for each query with progress bar
    all_metrics = []
    print(f"\n   Running {len(test_queries)} retrieval tests (top_k={top_k})...")
    
    for qa_pair in tqdm(test_queries, desc="   Testing retrieval", unit="query"):
        question = qa_pair["question"]
        relevant_chunk = {"text": qa_pair["passage"]}
        
        # Query vector DB
        retrieved_chunks = vdb.query(question, top_k=top_k)
        
        if len(retrieved_chunks) > 0:
            # Calculate metrics
            metrics = calculate_retrieval_metrics(
                retrieved_chunks=retrieved_chunks,
                relevant_chunk=relevant_chunk,
                top_k=top_k
            )
            all_metrics.append(metrics)
    
    # Aggregate metrics
    if all_metrics:
        avg_metrics = {
            "precision@k": statistics.mean([m.get(f"precision@{top_k}", 0) for m in all_metrics]),
            "recall@k": statistics.mean([m.get(f"recall@{top_k}", 0) for m in all_metrics]),
            "mrr": statistics.mean([m.get("mrr", 0) for m in all_metrics]),
            "hit_rate@k": statistics.mean([m.get(f"hit_rate@{top_k}", 0) for m in all_metrics]),
            "num_queries": len(all_metrics)
        }
        
        print(f"\n   ✅ Retrieval Test Results:")
        print(f"      Precision@{top_k}: {avg_metrics['precision@k']:.3f}")
        print(f"      Recall@{top_k}: {avg_metrics['recall@k']:.3f}")
        print(f"      MRR: {avg_metrics['mrr']:.3f}")
        print(f"      Hit Rate@{top_k}: {avg_metrics['hit_rate@k']:.3f}")
        print(f"      Number of queries: {avg_metrics['num_queries']}")
        
        return avg_metrics
    else:
        print("⚠️  No metrics calculated (no retrieval results)")
        return {}


def main():
    """Main function to demonstrate the workflow."""
    print("=" * 60)
    print("ChromaDB Retrieval Test")
    print("=" * 60)
    
    if not CHROMADB_AVAILABLE:
        print("❌ ChromaDB is not installed. Install with: pip install chromadb")
        return
    
    # Configuration
    CHROMADB_DIR = "./tests/assets/test_vdbs/chroma"  # Directory containing existing ChromaDB
    COLLECTION_NAME = None  # None = use first available collection
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # Version configuration (optional)
    SAVE_TO_VERSION = True  # Set to True to save results to version folder
    VERSION_NAME = None  # None = auto-generate from embedding model name
    VERSION_DESCRIPTION = ""  # Optional description for the version
    
    # Step 1: Load existing ChromaDB
    print("\n" + "="*60)
    print("STEP 1: Loading Existing ChromaDB")
    print("="*60)
    
    try:
        connection = load_existing_chromadb(
            persist_directory=CHROMADB_DIR,
            collection_name=COLLECTION_NAME,
            embedding_model_name=EMBEDDING_MODEL
        )
    except Exception as e:
        print(f"❌ Failed to load ChromaDB: {e}")
        return
    
    # Step 2: Get chunks (default: 10)
    NUM_CHUNKS = 300  # Parameter for number of chunks to retrieve
    print("\n" + "="*60)
    print(f"STEP 2: Retrieving {NUM_CHUNKS} Chunks")
    print("="*60)
    chunks = get_chunks_from_collection(connection, num_chunks=NUM_CHUNKS)
    
    if len(chunks) == 0:
        print("❌ No chunks retrieved. Collection may be empty.")
        return
    
    # Display sample chunks
    print(f"\n   Sample chunks (first 3):")
    for i, chunk in enumerate(chunks[:3]):
        text_preview = chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"]
        print(f"\n   Chunk {i+1}:")
        print(f"      ID: {chunk.get('id', 'N/A')}")
        print(f"      Text: {text_preview}")
        if chunk.get('metadata'):
            print(f"      Metadata: {chunk['metadata']}")
    
    # Step 3: Create pandas DataFrame with chunks, questions, answers, and retrieved docs
    print("\n" + "="*60)
    print("STEP 3: Creating Retrieval DataFrame")
    print("="*60)
    
    if not PANDAS_AVAILABLE:
        print("⚠️  pandas not available, skipping DataFrame creation")
        df = None
    else:
        try:
            df = create_retrieval_dataframe(
                connection=connection,
                chunks=chunks,
                top_k=5,
                generate_qa=True,  # Set to False to use simple queries instead
                device=None,  # Auto-detect device
                batch_size=8
            )
            
            # Display sample of DataFrame
            print(f"\n   DataFrame sample (first 3 rows):")
            print(df[['chunk_id', 'question', 'answer', 'num_retrieved']].head(3).to_string())
            
            # Save DataFrame to CSV
            output_file = "retrieval_results.csv"
            df.to_csv(output_file, index=False)
            print(f"\n   ✅ Saved DataFrame to {output_file}")
            
            # Save to version if enabled
            if SAVE_TO_VERSION:
                try:
                    version_name = VERSION_NAME or EMBEDDING_MODEL.replace("/", "-").replace("_", "-")
                    print(f"\n   Saving to version: {version_name}")
                    
                    # Create version
                    version_path = create_version(
                        version_name=version_name,
                        description=VERSION_DESCRIPTION or f"Evaluation with {EMBEDDING_MODEL}",
                        embedding_model=EMBEDDING_MODEL,
                        top_k=5
                    )
                    
                    # Prepare QA pairs for saving
                    qa_pairs = []
                    for idx, row in df.iterrows():
                        qa_pairs.append({
                            "question": row.get("question", ""),
                            "answer": row.get("answer", ""),
                            "passage": row.get("chunk", ""),
                            "chunk_id": row.get("chunk_id", "")
                        })
                    
                    # Prepare chunks for saving
                    chunks_list = []
                    for idx, row in df.iterrows():
                        chunks_list.append({
                            "id": row.get("chunk_id", ""),
                            "text": row.get("chunk", ""),
                            "metadata": {}
                        })
                    
                    # Save to version
                    save_to_version(
                        version_name=version_name,
                        qa_pairs=qa_pairs,
                        chunks=chunks_list,
                        results_df=df,
                        metadata={"top_k": 5}
                    )
                    
                    print(f"   ✅ Saved to version folder: {version_path}")
                except Exception as e:
                    print(f"   ⚠️  Failed to save to version: {e}")
                    import traceback
                    traceback.print_exc()
            
        except Exception as e:
            print(f"⚠️  Failed to create DataFrame: {e}")
            import traceback
            traceback.print_exc()
            df = None
    
    # Step 4: Test retrieval metrics
    print("\n" + "="*60)
    print("STEP 4: Testing Retrieval Metrics")
    print("="*60)
    retrieval_results = test_retrieval(
        connection=connection,
        num_test_queries=NUM_CHUNKS,
        top_k=5,
        chunks_for_testing=chunks[:NUM_CHUNKS]  # Use first NUM_CHUNKS chunks for testing
    )
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✅ Loaded ChromaDB from: {CHROMADB_DIR}")
    print(f"✅ Collection: {connection.collection_name}")
    try:
        print(f"✅ Total documents: {connection.collection.count()}")
    except Exception:
        print(f"✅ Total documents: (unavailable)")
    print(f"✅ Retrieved chunks: {len(chunks)}")
    if df is not None:
        print(f"✅ DataFrame created with {len(df)} rows")
        print(f"   DataFrame saved to: retrieval_results.csv")
    if retrieval_results:
        print(f"✅ Retrieval test completed")
        print(f"   Average Precision@5: {retrieval_results.get('precision@k', 0):.3f}")
        print(f"   Average Recall@5: {retrieval_results.get('recall@k', 0):.3f}")
        print(f"   Average MRR: {retrieval_results.get('mrr', 0):.3f}")
        print(f"   Average Hit Rate@5: {retrieval_results.get('hit_rate@k', 0):.3f}")
    print("="*60)
    df.to_csv("retrieval_results.csv", index=False)
    # Return DataFrame if created
    return df


if __name__ == "__main__":
    main()
