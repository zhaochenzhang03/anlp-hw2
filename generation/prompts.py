"""
Prompt templates for the reader. Bump PROMPT_VERSION whenever a template
changes, so saved predictions say which prompt produced them.
"""

PROMPT_VERSION = "v1"

# Style examples (not from the public dev set) showing the expected answer
# length: a bare name, number, date or short phrase.
STYLE_EXAMPLES = """Examples of the expected answer style:
Question: Who is Pittsburgh named after?
Answer: William Pitt
Question: In what year was the University of Pittsburgh founded?
Answer: 1787
Question: What is the accession number of the painting "Example Title"?
Answer: 70.12.3"""

RAG_SYSTEM = f"""You answer questions about Pittsburgh and Carnegie Mellon University (CMU) using the numbered context passages provided.

Rules:
- Reply with only the answer: a name, number, date or short phrase. Do not write a full sentence and do not explain.
- Copy names, numbers, dates, times and identifiers (such as accession numbers) exactly as they appear in the context.
- Base the answer on the context. If the context does not contain the answer, answer from your own knowledge only if you are confident; otherwise reply exactly: I don't know

{STYLE_EXAMPLES}"""

CLOSED_BOOK_SYSTEM = f"""You answer questions about Pittsburgh and Carnegie Mellon University (CMU).

Rules:
- Reply with only the answer: a name, number, date or short phrase. Do not write a full sentence and do not explain.
- If you do not know the answer, reply exactly: I don't know

{STYLE_EXAMPLES}"""

# Closed-book variant that always answers, so closed-book scores are not
# dominated by abstentions.
CLOSED_BOOK_GUESS_SYSTEM = f"""You answer questions about Pittsburgh and Carnegie Mellon University (CMU).

Rules:
- Reply with only the answer: a name, number, date or short phrase. Do not write a full sentence and do not explain.
- Always give your best guess, even if you are unsure.

{STYLE_EXAMPLES}"""


def format_context(chunks: list[dict]) -> str:
    """
    Number the retrieved chunks, each with its title and source URL.
    """
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        header = f"[{i}] {chunk.get('title', '').strip()}"
        if chunk.get("url"):
            header += f" ({chunk['url']})"
        blocks.append(f"{header}\n{chunk['text'].strip()}")
    return "\n\n".join(blocks)


def build_messages(question: str, chunks: list[dict] | None, always_answer: bool = False) -> list[dict]:
    """
    Chat messages for one question. chunks=None means closed-book;
    always_answer selects the closed-book prompt that forbids abstaining.
    """
    if chunks is None:
        return [
            {"role": "system", "content": CLOSED_BOOK_GUESS_SYSTEM if always_answer else CLOSED_BOOK_SYSTEM},
            {"role": "user", "content": f"Question: {question}\nAnswer:"},
        ]

    return [
        {"role": "system", "content": RAG_SYSTEM},
        {
            "role": "user",
            "content": f"Context:\n{format_context(chunks)}\n\nQuestion: {question}\nAnswer:",
        },
    ]
