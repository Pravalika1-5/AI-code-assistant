import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import json

# CONFIG
DEFAULT_REPO_URL = "https://github.com/parimienosh/Patient-Registration.git"
CLONE_DIR = "./cloned_repo"
DB_NAME = "chroma_repo_db"

# Use in-memory ChromaDB (works on Streamlit Cloud)
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="repo_files")


# -------- Read Static Files from Repo --------
# Instead of cloning, just read files from the current directory or skip
def read_all_files(repo_path):
    file_data = {}
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith((".java", ".xml", ".txt", ".md", ".json", ".yml")):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        file_data[path] = f.read()
                except Exception as e:
                    st.warning(f"Failed to read {path}: {e}")
    return file_data


def create_chroma_index(files_dict, model):
    for path, content in files_dict.items():
        embedding = model.encode(content).tolist()
        collection.add(ids=[path], embeddings=[embedding], metadatas=[{"path": path}])


def search_chroma(query, model):
    query_embedding = model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=1)
    if results and results["ids"] and results["ids"][0]:
        return results["ids"][0][0]
    return None


def ask_ollama_llm(file_content, user_question):
    return "⚠️ Ollama is not available on Streamlit Cloud. Please run this locally for full LLM features."


def ai_agent_interaction(files_dict):
    st.title("AI Code Assistant 🤖")
    st.write("Welcome! I am your AI Code Assistant.")

    menu = st.radio("Choose a task", ["Search Files", "Review Code", "Generate Test Cases", "Generate Code Summary"])

    if menu == "Search Files":
        query = st.text_input("Search something in the repository")
        if query and st.button("🔍 Search"):
            model = SentenceTransformer('all-MiniLM-L6-v2')
            best_match = search_chroma(query, model)
            if best_match:
                st.session_state["selected_file"] = best_match
                st.session_state["selected_content"] = files_dict[best_match]
                st.write(f"Found in: `{best_match}`")
                st.code(files_dict[best_match][:1000])

    elif menu in ["Review Code", "Generate Test Cases", "Generate Code Summary"]:
        selected_file = st.selectbox("Select a file:", list(files_dict.keys()))
        if selected_file:
            prompt_map = {
                "Review Code": "Analyze and provide a detailed review.",
                "Generate Test Cases": "Generate JUnit test cases.",
                "Generate Code Summary": "Give a high-level summary.",
            }
            st.write(f"Processing `{selected_file}`...")
            output = ask_ollama_llm(files_dict[selected_file], prompt_map[menu])
            st.write(output)


# Streamlit UI
st.set_page_config(page_title="AI Code Assistant", layout="wide")

st.info("This demo reads repo files and helps analyze them. LLM functionality works locally only.")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Read files only from a limited local folder (safe on Streamlit Cloud)
files_dict = read_all_files(".")
if not collection.count():
    create_chroma_index(files_dict, model)

ai_agent_interaction(files_dict)
