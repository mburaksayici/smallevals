import chromadb
from sentence_transformers import SentenceTransformer
from smallevals import SmallEvalsVDBConnection, evaluate_retrievals


def main():
    CHROMADB_DIR = "./tests/assets/test_vdbs/chroma"
    COLLECTION_NAME = None
    EMBEDDING_MODEL = "intfloat/e5-small-v2"
    
    print("=" * 60)
    print("ChromaDB Retrieval Evaluation")
    print("=" * 60)
    
    embedding = SentenceTransformer(EMBEDDING_MODEL)
    chroma_client = chromadb.PersistentClient(path=CHROMADB_DIR)
    
    if COLLECTION_NAME is None:
        collections = chroma_client.list_collections()
        COLLECTION_NAME = collections[0].name
    
    smallevals_vdb = SmallEvalsVDBConnection(
        connection=chroma_client,
        collection=COLLECTION_NAME,
        embedding=embedding
    )
    
    print(f"Connected to ChromaDB collection: {COLLECTION_NAME}")
    
    smallevals_result = evaluate_retrievals(
        connection=smallevals_vdb,
        top_k=10,
        n_chunks=200,
        device=None,
        results_folder=None
    )
    
    print(f"\nResults saved to: {smallevals_result['results_path']}")
    return smallevals_result


if __name__ == "__main__":
    main()
