import argparse
import os
from typing import List, Dict
from flask import Flask, request, jsonify
import requests
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import platform

def set_google_api_key():
    api_key = "AIzaSyD7VrRJrSa3W7u0syiZpWldChRCTiWLp-4"
    if platform.system() == "Windows":
        os.environ["GOOGLE_API_KEY"] = api_key
    else:  # Assuming macOS or Linux
        os.environ["GOOGLE_API_KEY"] = api_key
set_google_api_key()

# Initialize Flask app
app = Flask(__name__)

# Now you can retrieve the API key from the environment variable when needed
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

# Initialize history storage
session_history: Dict[str, List[Dict[str, str]]] = {}

# Load Huggingface model for embeddings
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Initialize Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")


def build_combined_prompt(query: str, context: List[str], history: List[Dict[str, str]]) -> str:
    base_prompt = (
        """
            I am going to ask you a question, which I would like you to answer strictly
            based on the given rules. 
            If the question or the intent involves any of the following actions, return the action name alone, you should not say anything else. your response must be only the action name phrase: 
            action names : create_order, cancel_order, collect_payment, view_invoice.
            Here are some examples:
            1. 'Create an order to book a bike' , say "create_order"
            2. 'Cancel my order', say"cancel_order"
            3. 'I want to pay for my purchase', say "collect_payment"
            4. 'Show me the invoice for order #123', say "view_invoice"
            Else, If the question does not involves any of the following actions, you need answer strictly based on the given context
            and If there is not enough information in the context to answer the question,
            then try to make a guess, based on the context.
            """
    )

    user_prompt = f" The question is '{query}'. Here is all the context you have: {' '.join(context)}"
    history_prompt = "\n".join([f"User: {item['query']}\nBot: {item['response']}" for item in history])
    
    return f"{base_prompt} {history_prompt} {user_prompt}"

def fetch_and_call_api(query):
    response = None
    response_data = {}  # Initialize response_data to an empty dictionary
    
    try:
        # Replace with your actual API endpoint
        response = requests.get("https://dummyapi.com/api", params={"query": query})
        response_data = response.json()
    except Exception as e:
        # Handle exception and print a friendly message
        return(f"Need API Key to call, to perform the action. ")

    if response and response_data.get("status") == "success":
        return(response_data["message"])

def execute_action(action_name: str) -> str:
    if action_name == "create_order":
        fetch_response = fetch_and_call_api("YOUR_API_KEY")
        return fetch_response + "Order created successfully."
    elif action_name == "cancel_order":
        fetch_response = fetch_and_call_api("YOUR_API_KEY")
        return fetch_response + "Order created successfully."
    elif action_name == "collect_payment":
        fetch_response = fetch_and_call_api("YOUR_API_KEY")
        return fetch_response + "Order created successfully."
    elif action_name == "view_invoice":
        fetch_response = fetch_and_call_api("YOUR_API_KEY")
        return fetch_response + "Order created successfully."
    else:
        return "No action taken."

def get_gemini_response(query: str, context: List[str], session_id: str) -> str:
    history = session_history.get(session_id, [])

    # Build the combined prompt
    prompt = build_combined_prompt(query, context, history)

    # Get response from Gemini
    response = model.generate_content(prompt)

    response_text = response.text.strip().lower()

    # Check if Gemini returned an action instead of a regular answer
    if response_text in ["create_order", "cancel_order", "collect_payment", "view_invoice"]:
        action_response = execute_action(response_text)
        
        # Append the action execution to session history
        session_history.setdefault(session_id, []).append({"query": query, "response": action_response})
        return action_response

    # Save the query and response in session history
    session_history.setdefault(session_id, []).append({"query": query, "response": response.text})

    return response.text


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_id = request.headers.get('x-user-id')
    session_id = request.headers.get('x-session-id')
    query = data.get('query')

    if not query:
        return jsonify({"error": "Query is required."}), 400

    try:
        # Query the collection to get relevant documents from ChromaDB
        context_results = collection.query(query_texts=[query], n_results=5, include=["documents", "metadatas"])

        # Flatten the nested structure of documents
        context = [doc for sublist in context_results["documents"] for doc in sublist]

        # Get the response from Gemini
        response = get_gemini_response(query, context, session_id)
        
        return jsonify({"bot_message": response})

    except Exception as e:
        # Log the error (you can also use a logging library for better error tracking)
        print(f"Error occurred: {e}")
        return jsonify({"error": "An internal error occurred. Please try again later."}), 500


def main(collection_name: str = "documents_collection", persist_directory: str = ".") -> None:
    global collection  # Declare the collection variable to access it in the chat function
    client = chromadb.PersistentClient(path=persist_directory)

    # Create embedding function using Huggingface transformers
    embedding_function = embedding_functions.HuggingFaceEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        api_key="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"  # Ensure you set this environment variable
    )

    # Get the collection
    collection = client.get_collection(name=collection_name, embedding_function=embedding_function)

    # Start Flask app
    app.run(port=10000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load documents from a directory into a Chroma collection")
    parser.add_argument("--persist_directory", type=str, default="chroma_storage", help="Directory to store the Chroma collection")
    parser.add_argument("--collection_name", type=str, default="documents_collection", help="Name of the Chroma collection")

    args = parser.parse_args()

    main(collection_name=args.collection_name, persist_directory=args.persist_directory)
