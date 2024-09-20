import argparse
import os
from typing import List, Dict

import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Initialize history storage
session_history: Dict[str, List[Dict[str, str]]] = {}

# Load Huggingface model for embeddings
#tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
#embedding_model = AutoModelForSeq2SeqLM.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Check if the GOOGLE_API_KEY environment variable is set. Prompt the user to set it if not.
google_api_key = None
if "GOOGLE_API_KEY" not in os.environ:
    google_api_key = input("Please enter your Google API Key: ")
    genai.configure(api_key=google_api_key)
else:
    google_api_key = os.environ["GOOGLE_API_KEY"]
    genai.configure(api_key=google_api_key)

# Initialize Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")


def build_prompt(query: str, context: List[str], history: List[Dict[str, str]]) -> str:
    """
    Builds a prompt for the Gemini LLM with session history.

    Args:
    query (str): The user's original query.
    context (List[str]): The context of the query, returned by embedding search.
    history (List[Dict[str, str]]): Previous query-response pairs for the session.

    Returns:
    A prompt string for the LLM.
    """
    base_prompt = (
        "I am going to ask you a question, which I would like you to answer"
        " based only on the provided context, and not any other information."
        " If there is not enough information in the context to answer the question,"
        ' say "I am not sure", then try to make a guess.'
        " Break your answer up into nicely readable paragraphs."
    )
    user_prompt = f" The question is '{query}'. Here is all the context you have:" f'{(" ").join(context)}'

    # Include session history
    history_prompt = "\n".join([f"User: {item['query']}\nBot: {item['response']}" for item in history])

    return f"{base_prompt} {history_prompt} {user_prompt}"


def get_gemini_response(query: str, context: List[str], session_id: str) -> str:
    """
    Queries the Gemini API to get a response to the user's question, with session history.

    Args:
    query (str): The user's query.
    context (List[str]): The context returned by the embedding search.
    session_id (str): Unique session identifier.

    Returns:
    A response from the Gemini LLM.
    """
    history = session_history.get(session_id, [])

    # Build the prompt with context and history
    prompt = build_prompt(query, context, history)

    # Get response from Gemini
    response = model.generate_content(prompt)

    # Save the query and response in session history
    session_history[session_id].append({"query": query, "response": response.text})

    return response.text


def main(collection_name: str = "documents_collection", persist_directory: str = ".") -> None:
    # Instantiate a persistent chroma client
    client = chromadb.PersistentClient(path=persist_directory)

    # Create embedding function using Huggingface transformers
    embedding_function = embedding_functions.HuggingFaceEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        api_key="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"
    )

    # Get the collection
    collection = client.get_collection(name=collection_name, embedding_function=embedding_function)

    # Simple input loop
    while True:
        # Get session ID and query from user input
        session_id = input("Session ID: ")
        if session_id not in session_history:
            session_history[session_id] = []

        query = input("Query: ")
        if len(query) == 0:
            print("Please enter a question. Ctrl+C to quit.\n")
            continue

        print("\nThinking...\n")

        # Query the collection to get the 5 most relevant results
        results = collection.query(query_texts=[query], n_results=5, include=["documents", "metadatas"])

        sources = "\n".join(
            [
                f"{result['filename']}: line {result['line_number']}"
                for result in results["metadatas"][0]  # type: ignore
            ]
        )

        # Get the response from Gemini
        response = get_gemini_response(query, results["documents"][0], session_id)  # type: ignore

        # Output the response with sources
        print(f"Response: {response}\n")
        print(f"Source documents:\n{sources}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load documents from a directory into a Chroma collection")

    parser.add_argument("--persist_directory", type=str, default="chroma_storage", help="Directory to store the Chroma collection")
    parser.add_argument("--collection_name", type=str, default="documents_collection", help="Name of the Chroma collection")

    # Parse arguments
    args = parser.parse_args()

    main(collection_name=args.collection_name, persist_directory=args.persist_directory)