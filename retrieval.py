import json
import re

import chromadb

from sentence_transformers import SentenceTransformer

from config import (
    PROCESSED_DIR,
    VECTOR_DB_DIR,
)


EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

COLLECTION_NAME = "maintai_manuals"


def load_chunks():

    chunks_file = (
        PROCESSED_DIR
        / "maintai_chunks.json"
    )

    with open(
        chunks_file,
        "r",
        encoding="utf-8",
    ) as file:

        chunks = json.load(file)

    return chunks


def create_vector_database():

    print("=" * 70)
    print("CREATING VECTOR DATABASE")
    print("=" * 70)

    chunks = load_chunks()

    print(
        f"Loaded chunks: {len(chunks)}"
    )

    print(
        f"Loading embedding model: "
        f"{EMBEDDING_MODEL_NAME}"
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    VECTOR_DB_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_DIR)
    )

    # Delete old collection if it exists
    try:
        client.delete_collection(
            COLLECTION_NAME
        )
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME
    )

    texts = []
    ids = []
    metadatas = []

    for chunk in chunks:

        texts.append(
            chunk["text"]
        )

        ids.append(
            chunk["chunk_id"]
        )

        metadata = {
            "device_id":
                chunk["device_id"],

            "device":
                chunk["device"],

            "manufacturer":
                chunk["manufacturer"],

            "page":
                chunk["page"],

            "section":
                chunk["section"],

            "chunk_type":
                chunk["chunk_type"],

            "error_code":
                (
                    chunk["error_code"]
                    if chunk["error_code"]
                    is not None
                    else ""
                ),
        }

        metadatas.append(
            metadata
        )

    print(
        "Creating embeddings..."
    )

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    print(
        "Saving to ChromaDB..."
    )

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings.tolist(),
    )

    print()

    print(
        f"Stored vectors: "
        f"{collection.count()}"
    )

    print(
        f"Database path: "
        f"{VECTOR_DB_DIR}"
    )

    print("=" * 70)

def semantic_search(
    query,
    device_id,
    top_k=5,
):

    collection = get_or_create_collection()

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k,
        where={
            "device_id": device_id
        },
    )

    return results


def test_semantic_search():

    query = (
        "The monitor turns on "
        "but the screen is blank"
    )

    results = semantic_search(
        query=query,
        device_id="philips_g40",
        top_k=5,
    )

    print("=" * 70)
    print("SEMANTIC SEARCH TEST")
    print("=" * 70)

    print(f"Query: {query}")
    print()

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for index, (
        document,
        metadata,
        distance,
    ) in enumerate(
        zip(
            documents,
            metadatas,
            distances,
        ),
        start=1,
    ):

        print("-" * 70)

        print(f"Result {index}")

        print(
            f"Device: "
            f"{metadata['device']}"
        )

        print(
            f"Page: "
            f"{metadata['page']}"
        )

        print(
            f"Section: "
            f"{metadata['section']}"
        )

        print(
            f"Distance: "
            f"{distance}"
        )

        print()

        print(document)

        print()


def exact_error_search(
    error_code,
    device_id,
):

    collection = get_or_create_collection()

    return collection.get(
        where={
            "$and": [
                {
                    "device_id": device_id
                },
                {
                    "error_code": str(
                        error_code
                    )
                },
            ]
        }
    )


def test_exact_error_search():

    error_code = "37"

    results = exact_error_search(
        error_code=error_code,
        device_id="servo_ventilator",
    )

    print("=" * 70)
    print("EXACT ERROR SEARCH TEST")
    print("=" * 70)

    print(
        f"Error code: {error_code}"
    )

    print()

    if not results["ids"]:

        print(
            "No matching error code found."
        )

        return

    for index, (
        document,
        metadata,
    ) in enumerate(
        zip(
            results["documents"],
            results["metadatas"],
        ),
        start=1,
    ):

        print("-" * 70)

        print(
            f"Result {index}"
        )

        print(
            f"Device: "
            f"{metadata['device']}"
        )

        print(
            f"Page: "
            f"{metadata['page']}"
        )

        print(
            f"Section: "
            f"{metadata['section']}"
        )

        print(
            f"Error code: "
            f"{metadata['error_code']}"
        )

        print()

        print(document)

        print()


def detect_error_code(query):

    patterns = [
        r"\berror\s+code\s+(\d{1,5})\b",
        r"\berror\s+(\d{1,5})\b",
        r"\bcode\s+(\d{1,5})\b",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            query,
            re.IGNORECASE,
        )

        if match:
            return match.group(1)

    return None

def retrieve(
    query,
    device_id,
    error_code=None,
    top_k=5,
):

    # -------------------------------------
    # Detect error code automatically
    # -------------------------------------

    if error_code is None:

        error_code = detect_error_code(
            query
        )

    # -------------------------------------
    # Exact error-code search
    # -------------------------------------

    if error_code:

        results = exact_error_search(
            error_code=error_code,
            device_id=device_id,
        )

        # If exact code exists, use it
        if results["ids"]:

            return {
                "retrieval_type":
                    "exact_error",

                "detected_error_code":
                    error_code,

                "results":
                    results,
            }

    # -------------------------------------
    # Otherwise semantic search
    # -------------------------------------

    results = semantic_search(
        query=query,
        device_id=device_id,
        top_k=top_k,
    )

    return {
        "retrieval_type":
            "semantic",

        "detected_error_code":
            error_code,

        "results":
            results,
    }


def test_retrieve():

    print("=" * 70)
    print("RETRIEVE TEST")
    print("=" * 70)

    # =====================================
    # Test 1: Exact error code
    # =====================================

    print("\nTEST 1: EXACT ERROR SEARCH\n")

    exact = retrieve(
        query="What does error code 37 mean?",
        device_id="servo_ventilator",
        error_code="37",
    )

    print(
        f"Retrieval type: "
        f"{exact['retrieval_type']}"
    )

    exact_results = exact["results"]

    if exact_results["ids"]:

        print(
            exact_results["documents"][0]
        )

    else:

        print(
            "No exact error code found."
        )


    # =====================================
    # Test 2: Semantic search
    # =====================================

    print("\n")
    print("=" * 70)
    print("TEST 2: SEMANTIC SEARCH")
    print("=" * 70)
    print()

    semantic = retrieve(
        query=(
            "The monitor turns on "
            "but the screen is blank"
        ),
        device_id="philips_g40",
        top_k=3,
    )

    print(
        f"Retrieval type: "
        f"{semantic['retrieval_type']}"
    )

    semantic_results = (
        semantic["results"]
    )

    documents = (
        semantic_results["documents"][0]
    )

    metadatas = (
        semantic_results["metadatas"][0]
    )

    distances = (
        semantic_results["distances"][0]
    )

    for index, (
        document,
        metadata,
        distance,
    ) in enumerate(
        zip(
            documents,
            metadatas,
            distances,
        ),
        start=1,
    ):

        print("-" * 70)

        print(
            f"Result {index}"
        )

        print(
            f"Device: "
            f"{metadata['device']}"
        )

        print(
            f"Page: "
            f"{metadata['page']}"
        )

        print(
            f"Section: "
            f"{metadata['section']}"
        )

        print(
            f"Distance: "
            f"{distance}"
        )

        print()

        print(document)

        print()  

def test_auto_error_detection():

    query = "What does error 37 mean?"

    result = retrieve(
        query=query,
        device_id="servo_ventilator",
    )

    print("=" * 70)
    print("AUTO ERROR DETECTION TEST")
    print("=" * 70)

    print(f"Query: {query}")

    print(
        f"Detected error code: "
        f"{result['detected_error_code']}"
    )

    print(
        f"Retrieval type: "
        f"{result['retrieval_type']}"
    )

    print()

    if result["retrieval_type"] == "exact_error":

        for document in result["results"]["documents"]:

            print(document)

if __name__ == "__main__":

    query = "There is no CO2 readings."

    results = semantic_search(
        query=query,
        device_id="sc6002xl",
        top_k=10,
    )

    print("=" * 70)
    print("SEMANTIC SEARCH DEBUG")
    print("=" * 70)

    for i in range(len(results["documents"][0])):

        print()
        print(f"RESULT {i + 1}")
        print("-" * 70)

        print(
            "Distance:",
            results["distances"][0][i]
        )

        print(
            "Metadata:",
            results["metadatas"][0][i]
        )

        print()

        print(
            results["documents"][0][i]
        )

import json
import chromadb
from sentence_transformers import SentenceTransformer

from config import PROCESSED_DIR, VECTOR_DB_DIR


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "maintai_manuals"


model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)


def get_or_create_collection():

    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_DIR)
    )

    try:

        collection = client.get_collection(
            name=COLLECTION_NAME
        )

        return collection

    except Exception:

        print(
            "Vector collection not found. "
            "Creating it from processed chunks..."
        )

        chunks_path = (
            PROCESSED_DIR
            / "maintai_chunks.json"
        )

        with open(
            chunks_path,
            "r",
            encoding="utf-8",
        ) as file:

            chunks = json.load(file)

        documents = []
        ids = []
        metadatas = []

        for chunk in chunks:

            ids.append(
                chunk["id"]
            )

            documents.append(
                chunk["text"]
            )

            metadatas.append(
                {
                    "device_id": chunk[
                        "device_id"
                    ],
                    "device": chunk[
                        "device"
                    ],
                    "manufacturer": chunk[
                        "manufacturer"
                    ],
                    "page": chunk[
                        "page"
                    ],
                    "section": chunk[
                        "section"
                    ],
                    "chunk_type": chunk[
                        "chunk_type"
                    ],
                    "error_code": (
                        chunk.get(
                            "error_code"
                        )
                        or ""
                    ),
                }
            )

        embeddings = model.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        collection = client.create_collection(
            name=COLLECTION_NAME
        )

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=[
                embedding.tolist()
                for embedding
                in embeddings
            ],
        )

        print(
            f"Created collection with "
            f"{len(ids)} chunks."
        )

        return collection