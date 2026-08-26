import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


MODEL_NAME = "openai/gpt-oss-20b"


client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def generate_answer(
    query,
    context,
):

   system_prompt = """
You are Fixora, a technical maintenance assistant.

Your job is to answer the user's question using only the
provided service-manual evidence.

Rules:

1. Use only the provided manual evidence.

2. Do not invent causes, procedures, measurements, or steps.

3. If the manual evidence is insufficient, clearly say that.

4. Preserve technical terminology from the manual.

5. Preserve the order and relationships found in the provided evidence.
Do not invent a troubleshooting priority or sequence unless the manual
explicitly provides one.

6. Mention the manual page and section when available.

7. Do not claim something is confirmed unless the evidence confirms it.

8. If the manual gives multiple possible causes, present them as possibilities.

9. Keep the answer practical and concise, but do not add procedural wording
that is not present in the evidence.

10. Do not add an "order of checks", priority, diagnosis, or recommendation
unless that ordering is explicitly supported by the provided evidence.

11. If multiple retrieved chunks describe different possible causes for the
same symptom, present them as separate possible causes without ranking them.

12. Do not tell the user to check causes "in turn", "first", "next", or in any
sequence unless the manual explicitly provides that sequence.
13. End the answer after presenting the supported causes, actions, and references.
Do not add a concluding instruction unless that instruction is explicitly present
in the manual evidence.
"""

   user_prompt = f"""
USER QUESTION:

{query}


SERVICE MANUAL EVIDENCE:

{context}


Answer the question using only the evidence above.
"""

   response = client.chat.completions.create(
        model=MODEL_NAME,

        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],

        temperature=0.2,
    )

   return (
        response
        .choices[0]
        .message
        .content
    )