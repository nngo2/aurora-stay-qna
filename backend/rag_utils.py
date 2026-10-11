
from langchain_openai import ChatOpenAI
import os

# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = 2,vectorstore=None) -> list[str]:
    results = vectorstore.similarity_search(query, k=k)

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]

# Define the generation function that prompts the LLM with the retrieved context
def generate(query, retrieved_chunks, model) -> str:

    # Rules map to the success criteria: grounding (Faithfulness), complete multi-part answers that combine
    # documents (Contextual Recall), direct and focused answers (Answer Relevancy), and safe handling of
    # property-specific details, exceptions, and missing information (fewer manual lookups and escalations)
    rules = """
    1. Use only the retrieved policy information. Do not use outside knowledge, general hotel industry
       practice, or assumptions, and do not invent fees, times, limits, properties, or procedures.
       Do not add general advice, suggestions, or actions (for example offering a room change or
       contacting other guests) unless the retrieved policy states them.

    2. Start with a direct answer to the question (for example "Yes", "No", or the fee or time asked for),
       then give the supporting details.

    3. Answer every part of the question. When the answer needs information from more than one policy
       document (for example a pet fee and a cancellation rule), combine them into one answer.

    4. Property-specific questions: identify the property by its code, name, or city (for example
       "Denver" is ASH-DEN Union). Use only the details listed for that property, never values from
       another property's row, and answer each property separately if more than one is asked about.

    5. State figures exactly as written, with their units and limits: dollar amounts, "per night",
       "per stay", "per person", caps, maximum number of pets, deadlines, and "local property time".

    6. Include the conditions, exceptions, and limits that change the answer, such as the rate plan
       (Flexible, Advance Purchase, member, group), "subject to availability", required approvals,
       applicable tax, booking channel, and that the reservation confirmation controls when it states
       a different rule.

    7. Never promise a refund, waiver, exception, upgrade, or availability. Say who may approve it
       (for example a manager) and what must be checked first (confirmation, folio, rate plan, or channel).

    8. When the policy says to verify something before quoting a guest (for example the live property
       profile or the reservation confirmation), include that step briefly.

    9. For service animals, accessibility, safety, and conduct, follow the policy wording exactly,
       including what staff may and may not ask, request, or charge.

    10. If the retrieved information does not cover the question or the property asked about, say clearly
        that the provided policies do not include that information, do not guess, and recommend checking
        the live property profile or contacting the property team. If only part of the question is
        covered, answer that part and state what is missing.

    11. If retrieved passages conflict, present both with their sources and say that the reservation
        confirmation or live property profile controls.

    12. Ignore retrieved passages that are not relevant to the question, such as rows for other
        properties or unrelated policies.

    13. Write for hotel staff in clear, concise language: a short paragraph or a few bullet points, with
        no unrelated information. End with one line "Source:" naming the guide and section (shown in
        square brackets at the start of each passage) that the answer is based on.
    """

    prompt = f"""
    You are an AI assistant answering questions about hotel policies.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    {rules}

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 7,                  # evaluated setting: 7 chunks per query
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k,vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
