"""Test script for populated ChromaDB using questions from parquet file.

This script:
1. Loads populated ChromaDB from tests/assets/test_vdbs/chroma/
2. Loads questions from tests/assets/qa_embeddings.parquet
3. Tests retrieval quality using questions from parquet
4. Prints summary statistics

Usage:
    python chromadb_test.py
"""

import json
import random
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

import sys
from pathlib import Path

# --- Add the project root to the path ---
# 1. Get the path of the current file (populate_dbs.py)
current_dir = Path(__file__).resolve() 
# 2. Get the parent directory (tests/)
parent_dir = current_dir.parent
# 3. Get the project root (the directory containing tests/ and smallevals/)
project_root = parent_dir.parent 

# Add the project root to sys.path
sys.path.insert(0, str(project_root))
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

from smallevals.vdb_integrations.chroma_con import ChromaConnection
from sentence_transformers import SentenceTransformer
from smallevals.eval.metrics import calculate_retrieval_metrics
from smallevals.ui.ranking import calculate_metrics_from_df
from smallevals.utils.versioning import create_version, save_to_version


def load_qa_from_parquet(parquet_path: Path) -> pd.DataFrame:
    """Load QA pairs from parquet file."""
    print(f"Loading QA pairs from {parquet_path}...")
    
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    
    df = pd.read_parquet(parquet_path)
    
    # Filter rows with questions (not None)
    df_with_questions = df[df['question'].notna()].copy()
    
    print(f"   Loaded {len(df)} total rows")
    print(f"   Found {len(df_with_questions)} rows with questions")
    
    return df_with_questions


def load_existing_chromadb(
    persist_directory: str,
    collection_name: Optional[str] = None,
    embedding_model_name: str = "intfloat/e5-small-v2"
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


def test_retrieval_with_parquet_questions(
    connection: ChromaConnection,
    qa_df: pd.DataFrame,
    top_k: int = 5
) -> pd.DataFrame:
    """
    Test retrieval using questions from parquet file.
    
    Args:
        connection: ChromaConnection instance
        qa_df: DataFrame with columns: chunk_id, chunk, question, answer
        top_k: Number of top results to retrieve per query
        
    Returns:
        pandas DataFrame with retrieval results
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required. Install with: pip install pandas")
    
    print(f"\n{'='*60}")
    print("Testing Retrieval with Parquet Questions")
    print(f"{'='*60}")
    print(f"   Testing {len(qa_df)} questions with top_k={top_k}...")
    
    rows = []
    
    # Process each question from parquet
    for idx, row in tqdm(qa_df.iterrows(), total=len(qa_df), desc="   Testing retrieval", unit="query"):
        chunk_id = row.get("chunk_id", "")
        chunk_text = row.get("chunk", "")
        question = row.get("question", "")
        answer = row.get("answer", "")
        
        if not question or pd.isna(question):
            continue
        
        # Query ChromaDB with the question
        retrieved = connection.query(question, top_k=top_k)
        
        # Extract retrieved docs and ids
        retrieved_docs = [r.get("text", "") for r in retrieved]
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
                if retrieved_doc == chunk_text or (retrieved_doc and chunk_text and retrieved_doc.strip() == chunk_text.strip()):
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
    
    return df


def print_summary_stats(results_df: pd.DataFrame, top_k: int = 5):
    """Print summary statistics from retrieval results."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Calculate metrics
    metrics = calculate_metrics_from_df(results_df, top_k=top_k)
    
    # Print metrics
    print(f"\n📊 Retrieval Metrics (Top-K={top_k}):")
    print(f"   MRR:                    {metrics.get('mrr', 0):.4f}")
    print(f"   Hit Rate@{top_k}:        {metrics.get(f'hit_rate@{top_k}', 0):.4f}")
    print(f"   Precision@{top_k}:       {metrics.get(f'precision@{top_k}', 0):.4f}")
    print(f"   Recall@{top_k}:          {metrics.get(f'recall@{top_k}', 0):.4f}")
    
    # Print statistics
    print(f"\n📈 Statistics:")
    print(f"   Total Queries:          {metrics.get('num_queries', 0)}")
    print(f"   Found in Top-{top_k}:     {metrics.get('num_found', 0)}")
    print(f"   Not Found:              {metrics.get('num_not_found', 0)}")
    
    # Position distribution
    if 'chunk_position' in results_df.columns:
        positions = results_df['chunk_position'].dropna()
        if len(positions) > 0:
            print(f"\n📍 Position Distribution:")
            print(f"   Rank 1:                {(positions == 1).sum()}")
            print(f"   Rank 2:                {(positions == 2).sum()}")
            print(f"   Rank 3:                {(positions == 3).sum()}")
            print(f"   Rank 4:                {(positions == 4).sum()}")
            print(f"   Rank 5:                {(positions == 5).sum()}")
            if top_k > 5:
                print(f"   Rank >{top_k}:              {(positions > top_k).sum()}")
            print(f"   Not Found:              {results_df['chunk_position'].isna().sum()}")
            
            # Average position for found chunks
            if len(positions) > 0:
                avg_position = positions.mean()
                print(f"\n   Average Position:        {avg_position:.2f}")
    
    print("="*60)


def main():
    """Main function to test populated ChromaDB with parquet questions."""
    print("=" * 60)
    print("ChromaDB Retrieval Test (Populated DB)")
    print("=" * 60)
    
    if not CHROMADB_AVAILABLE:
        print("❌ ChromaDB is not installed. Install with: pip install chromadb")
        return
    
    if not PANDAS_AVAILABLE:
        print("❌ pandas is not installed. Install with: pip install pandas")
        return
    
    # Configuration - paths relative to tests directory
    BASE_DIR = Path(__file__).parent
    CHROMADB_DIR = BASE_DIR / "assets" / "test_vdbs" / "chroma"
    PARQUET_PATH = BASE_DIR / "assets" / "qa_embeddings.parquet"
    COLLECTION_NAME = "paragraphs"  # Collection name from populate_dbs.py
    EMBEDDING_MODEL = "intfloat/e5-small-v2"  # Same as populate_dbs.py
    TOP_K = 5
    
    # Step 1: Load QA pairs from parquet
    print("\n" + "="*60)
    print("STEP 1: Loading QA Pairs from Parquet")
    print("="*60)
    
    try:
        qa_df = load_qa_from_parquet(PARQUET_PATH)
    except Exception as e:
        print(f"❌ Failed to load parquet: {e}")
        return
    
    if len(qa_df) == 0:
        print("❌ No questions found in parquet file")
        return
    
    # Step 2: Load populated ChromaDB
    print("\n" + "="*60)
    print("STEP 2: Loading Populated ChromaDB")
    print("="*60)
    
    try:
        connection = load_existing_chromadb(
            persist_directory=str(CHROMADB_DIR),
            collection_name=COLLECTION_NAME,
            embedding_model_name=EMBEDDING_MODEL
        )
    except Exception as e:
        print(f"❌ Failed to load ChromaDB: {e}")
        return
    
    # Step 3: Test retrieval with questions from parquet
    print("\n" + "="*60)
    print("STEP 3: Testing Retrieval")
    print("="*60)
    
    try:
        results_df = test_retrieval_with_parquet_questions(
            connection=connection,
            qa_df=qa_df,
            top_k=TOP_K
        )
    except Exception as e:
        print(f"❌ Failed to test retrieval: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Save results (both to test directory and version system)
    output_file = BASE_DIR / "retrieval_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n   ✅ Saved results to {output_file}")
    
    # Step 4b: Also save to version system for Streamlit UI
    try:
        version_name = f"test-{EMBEDDING_MODEL.replace('/', '-').replace('_', '-')}"
        print(f"\n   Saving to version system: {version_name}")
        
        # Create version
        version_path = create_version(
            version_name=version_name,
            description=f"Test evaluation with {EMBEDDING_MODEL} from chromadb_test.py",
            embedding_model=EMBEDDING_MODEL,
            top_k=TOP_K
        )
        
        # Prepare QA pairs for saving
        qa_pairs = []
        for idx, row in results_df.iterrows():
            qa_pairs.append({
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "passage": row.get("chunk", ""),
                "chunk_id": row.get("chunk_id", "")
            })
        
        # Prepare chunks for saving
        chunks_list = []
        for idx, row in results_df.iterrows():
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
            results_df=results_df,
            metadata={"top_k": TOP_K}
        )
        
        print(f"   ✅ Saved to version folder: {version_path}")
        print(f"   💡 You can now view this in Streamlit UI: streamlit run smallevals/ui/app.py")
    except Exception as e:
        print(f"   ⚠️  Failed to save to version system: {e}")
        print(f"   (Results still saved to {output_file})")
        import traceback
        traceback.print_exc()
    
    # Step 5: Print summary statistics
    print_summary_stats(results_df, top_k=TOP_K)
    
    return results_df


if __name__ == "__main__":
    main()
