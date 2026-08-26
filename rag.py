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
    """
    Convert semantic search results
    into clean RAG context.
    """

    if not results["documents"]:
        return ""

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    context_parts = []

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

        source = (
            f"SOURCE {index}\n"
            f"Device: {metadata['device']}\n"
            f"Page: {metadata['page']}\n"
            f"Section: {metadata['section']}\n"
            f"Distance: {distance:.4f}\n"
            f"Manual evidence:\n"
            f"{document}"
        )

        context_parts.append(source)

    return "\n\n".join(context_parts)


def build_rag_context(
    query,
    device_id,
    top_k=5,
):

    retrieval_output = retrieve(
        query=query,
        device_id=device_id,
        top_k=top_k,
    )

    retrieval_type = (
        retrieval_output[
            "retrieval_type"
        ]
    )

    results = retrieval_output[
        "results"
    ]

    if retrieval_type == "exact_error":

        context = format_exact_results(
            results
        )

    else:

        context = format_semantic_results(
            results
        )

    return {
        "retrieval_type":
            retrieval_type,

        "detected_error_code":
            retrieval_output.get(
                "detected_error_code"
            ),

        "context":
            context,

        "raw_results":
            results,
    }

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

def answer_query(
    query,
    device_id,
    top_k=5,
):

    rag_result = build_rag_context(
        query=query,
        device_id=device_id,
        top_k=top_k,
    )

    context = rag_result["context"]

    if not context:
        return {
            "answer":
                "No relevant information was found "
                "in the selected service manual.",

            "retrieval_type":
                rag_result["retrieval_type"],

            "context":
                "",
        }

    answer = generate_answer(
        query=query,
        context=context,
    )

    return {
        "answer":
            answer,

        "retrieval_type":
            rag_result["retrieval_type"],

        "context":
            context,
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