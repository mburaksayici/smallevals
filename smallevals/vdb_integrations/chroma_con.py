"""Chroma Connection to export Chonkie's Chunks into a Chroma collection."""

import importlib.util as importutil
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Union,
    cast,
)
import logging
from uuid import NAMESPACE_OID, uuid5

from .base import BaseVDBConnection
from .utils import generate_random_collection_name

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import chromadb
    import numpy as np
    from sentence_transformers import SentenceTransformer


# NOTE: This is a bit of a hack to work with Chroma's EmbeddingFunction interface
# since we can't have a EmbeddingFunction without having ChromaDB in the base install.
# So we create a local class (which we don't pass to our namespace) that mimics the
# interface of chromadb.EmbeddingFunction. It has a __call__ that takes in a input
# and returns a numpy array.

# Since chromadb.Documents and chromadb.Embeddings are just strings and numpy arrays respectively,
# we can just return the numpy array from __call__ and be done with it.


class ChromaEmbeddingFunction:
    """Chroma Embedding Function.

    Embeds the text of the chunks using the embedding model and
    adds the embeddings to the chunks for use in downstream tasks
    like upserting into a vector database.

    Args:
        embedding_model: The embedding model to use.
        **kwargs: Additional keyword arguments.

    """

    def __init__(
        self,
        embedding_model: "SentenceTransformer",
        **kwargs: Dict[str, Any],
    ) -> None:
        """Initialize the ChromaEmbeddingFunction."""
        self.embedding_model = embedding_model
        self._model_name = getattr(embedding_model, 'model_name', str(embedding_model))

    def name(self) -> str:
        """Return the name of the embedding model for ChromaDB compatibility."""
        return self._model_name

    def __call__(
        self, input: Union[str, List[str]]
    ) -> Union["np.ndarray", List["np.ndarray"]]:
        """Call the ChromaEmbeddingFunction."""
        if isinstance(input, str):
            return self.embedding_model.encode(input)
        elif isinstance(input, list):
            return self.embedding_model.encode(input)
        else:
            raise ValueError("Input must be a string or a list of strings.")


class ChromaConnection(BaseVDBConnection):
    """Chroma Connection to export Chonkie's Chunks into a Chroma collection.

    This handshake is experimental and may change in the future. Not all Chonkie features are supported yet.

    Args:
        client: The Chroma client to use.
        collection_name: The name of the collection to use.
        embedding_model: The embedding model to use.
        path: The path to the Chroma collection locally. If provided, it will create a Persistent Chroma Client.

    """

    def __init__(
        self,
        client: Optional[Any] = None,  # chromadb.Client
        collection_name: Union[str, Literal["random"]] = "random",
        embedding_model: Optional["SentenceTransformer"] = None,
        path: Optional[str] = None,
    ) -> None:
        """Initialize the Chroma Connection.

        Args:
            client: The Chroma client to use.
            collection_name: The name of the collection to use.
            embedding_model: Optional SentenceTransformer model (HuggingFace). If provided, will be used for encoding.
            path: The path to the Chroma collection locally. If provided, it will create a Persistent Chroma Client.

        """
        super().__init__()

        # Lazy importing the dependencies
        self._import_dependencies()

        # Initialize Chroma client
        if client is None and path is None:
            self.client = chromadb.Client()
        elif client is None and path is not None:
            self.client = chromadb.PersistentClient(path=path)
        else:
            self.client = client  # type: ignore[assignment]

        # Initialize the embedding function if model is provided
        if embedding_model is not None:
            self.embedding_function = ChromaEmbeddingFunction(embedding_model)
        else:
            self.embedding_function = None

        # If the collection name is not random, create the collection
        collection_kwargs = {}
        if self.embedding_function is not None:
            collection_kwargs["embedding_function"] = self.embedding_function  # type: ignore[arg-type]
        
        if collection_name != "random":
            self.collection_name = collection_name
            self.collection = self.client.get_or_create_collection(
                self.collection_name, **collection_kwargs
            )  # type: ignore[arg-type]
        else:
            # Keep generating random collection names until we find one that doesn't exist
            while True:
                self.collection_name = generate_random_collection_name()
                try:
                    self.collection = self.client.create_collection(
                        self.collection_name, **collection_kwargs
                    )  # type: ignore[arg-type]
                    break
                except Exception:
                    pass
            logger.info(f"Chonkie created a new collection in ChromaDB: {self.collection_name}")

        # Now that we have a collection, we can write the Chunks to it!

    def _is_available(self) -> bool:
        """Check if the dependencies are available."""
        return importutil.find_spec("chromadb") is not None

    def _import_dependencies(self) -> None:
        """Lazy import the dependencies."""
        if self._is_available():
            global chromadb
            import chromadb
        else:
            raise ImportError(
                "ChromaDB is not installed. "
                + "Please install it with `pip install chonkie[chroma]`."
            )

    def __repr__(self) -> str:
        """Return the string representation of the ChromaConnection."""
        return f"ChromaConnection(collection_name={self.collection_name})"

    def search(
        self,
        query: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search the Chroma collection for similar chunks.

        Args:
            query: The query string to search for. If provided, `embedding` is ignored.
            embedding: The embedding vector to search for.
            limit: The maximum number of results to return.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the matching chunks and their metadata.

        """
        logger.debug(f"Searching Chroma collection: {self.collection_name} with limit={limit}")
        
        # Determine the query embeddings
        if embedding is not None:
            query_embeddings = [embedding]  # type: ignore[list-item]
        elif query is not None:
            if self.embedding_function is None:
                raise ValueError("embedding_function must be provided to encode query strings")
            query_embedding_result = cast("np.ndarray", self.embedding_function(query))
            query_embeddings = [query_embedding_result.tolist()]
        else:
            raise ValueError("Either query or embedding must be provided")

        # Perform the query
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=limit,
            include=["metadatas", "documents", "distances"],
        )

        # Safely extract results, checking for None values
        ids_list = results.get("ids")
        distances_list = results.get("distances")
        metadatas_list = results.get("metadatas")
        documents_list = results.get("documents")

        # Ensure all required result lists are present and not None
        if (
            ids_list is None
            or distances_list is None
            or metadatas_list is None
            or documents_list is None
        ):
            return []

        # We queried with one vector, so we get the first list of results
        ids, distances, metadatas, documents = (
            ids_list[0],
            distances_list[0],
            metadatas_list[0],
            documents_list[0],
        )

        # Process and format the results
        matches = []
        distance_metric = (
            self.collection.metadata.get("hnsw:space", "l2")
            if self.collection.metadata
            else "l2"
        )

        for id_val, distance, metadata, document in zip(
            ids, distances, metadatas, documents
        ):
            similarity = None
            if distance is not None:
                if distance_metric == "cosine":
                    similarity = 1.0 - distance
                elif distance_metric == "l2":
                    similarity = 1.0 - (distance**2 / 2)
                else:  # 'ip' (inner product) is already a similarity score
                    similarity = distance

            match_data = {
                "id": id_val,
                "score": similarity,
                "text": document,
            }
            if metadata:
                match_data.update(metadata)

            matches.append(match_data)

        logger.info(f"Search complete: found {len(matches)} matching chunks")
        return matches

    def sample_chunks(self, num_chunks: int) -> List[Dict[str, Any]]:
        """Sample random chunks from the Chroma collection.

        Args:
            num_chunks: Number of chunks to sample

        Returns:
            List of dictionaries with chunk information.
        """
        import random

        # Step 1: Fetch only IDs (VERY LIGHTWEIGHT)
        all_ids = self.collection.get(include=[])["ids"]

        if not all_ids:
            return []

        # Step 2: Sample IDs
        sampled_ids = random.sample(all_ids, min(num_chunks, len(all_ids)))

        # Step 3: Fetch only sampled documents
        data = self.collection.get(ids=sampled_ids)

        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])
        ids = data.get("ids", [])

        # Step 4: Build output
        results = []
        for i in range(len(ids)):
            results.append({
                "text": documents[i] if i < len(documents) else "",
                "metadata": metadatas[i] if i < len(metadatas) and metadatas[i] else {},
                "id": ids[i],
            })

        return results
