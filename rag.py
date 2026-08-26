from retrieval import retrieve
from llm import generate_answer

def format_exact_results(results):
    """
    Convert exact error-code results
    into clean RAG context.
    """

    if not results["ids"]:
        return ""

    context_parts = []

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

        source = (
            f"SOURCE {index}\n"
            f"Device: {metadata['device']}\n"
            f"Page: {metadata['page']}\n"
            f"Section: {metadata['section']}\n"
            f"Error code: {metadata['error_code']}\n"
            f"Manual evidence:\n"
            f"{document}"
        )

        context_parts.append(source)

    return "\n\n".join(context_parts)


def format_semantic_results(results):

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not documents:
        return "No relevant manual evidence found."

    context_parts = []

    max_results = min(3, len(documents))

    for i in range(max_results):

        document = documents[i]
        metadata = metadatas[i]
        distance = distances[i]

        context_parts.append(
            f"""
SOURCE {i + 1}

Device: {metadata.get("device")}
Page: {metadata.get("page")}
Section: {metadata.get("section")}
Chunk type: {metadata.get("chunk_type")}
Distance: {distance:.4f}

Manual evidence:
{document}
""".strip()
        )

    return "\n\n".join(context_parts)

def test_rag_context():

    result = build_rag_context(
        query=(
            "The monitor turns on "
            "but the screen is blank"
        ),
        device_id="philips_g40",
        top_k=3,
    )

    print("=" * 70)
    print("RAG CONTEXT TEST")
    print("=" * 70)

    print(
        f"Retrieval type: "
        f"{result['retrieval_type']}"
    )

    print()

    print(
        result["context"]
    )

def build_rag_context(
    query,
    device_id,
    top_k=5,
):

    retrieval_data = retrieve(
        query=query,
        device_id=device_id,
        top_k=top_k,
    )

    retrieval_type = retrieval_data["retrieval_type"]

    if retrieval_type == "exact_error":

        context = format_exact_results(
            retrieval_data["results"]
        )

    else:

        context = format_semantic_results(
            retrieval_data["results"]
        )

    return {
        "retrieval_type": retrieval_type,
        "detected_error_code": retrieval_data[
            "detected_error_code"
        ],
        "context": context,
        "raw_results": retrieval_data["results"],
    }
def answer_query(
    query,
    device_id,
    top_k=5,
):

    rag_data = build_rag_context(
        query=query,
        device_id=device_id,
        top_k=top_k,
    )

    answer = generate_answer(
        query=query,
        context=rag_data["context"],
    )

    return {
        "display_text": answer["display_text"],
        "speech_text": answer["speech_text"],
        "retrieval_type": rag_data["retrieval_type"],
        "detected_error_code": rag_data[
            "detected_error_code"
        ],
    }
def test_full_rag():

    query = "What does error 37 mean?"

    result = answer_query(
        query=query,
        device_id="servo_ventilator",
    )

    print("=" * 70)
    print("FULL RAG TEST")
    print("=" * 70)

    print(
        f"Retrieval type: "
        f"{result['retrieval_type']}"
    )

    print()

    print("ANSWER:")
    print()

    print(
        result["answer"]
    )
if __name__ == "__main__":
  test_full_rag()