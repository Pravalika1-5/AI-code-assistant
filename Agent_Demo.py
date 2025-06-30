import os
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import json

# -------- CONFIG --------
DEFAULT_REPO_URL = "https://github.com/parimienosh/Patient-Registration.git"
CLONE_DIR = "./cloned_repo"

# In-memory storage
file_paths = []
embedding_matrix = None
faiss_index = None


# -------- Read Files --------
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


# -------- Create FAISS Index --------
def create_faiss_index(files_dict, model):
    global file_paths, embedding_matrix, faiss_index
    file_paths = list(files_dict.keys())
    embeddings = [model.encode(files_dict[path]) for path in file_paths]
    embedding_matrix = np.vstack(embeddings).astype("float32")

    faiss_index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    faiss_index.add(embedding_matrix)


# -------- Search --------
def search_faiss(query, model):
    if faiss_index is None:
        return None
    query_vec = model.encode(query).astype("float32").reshape(1, -1)
    distances, indices = faiss_index.search(query_vec, k=1)
    return file_paths[indices[0][0]] if indices[0][0] < len(file_paths) else None


# -------- LLM Stub --------
def ask_ollama_llm(file_content, user_question):
    return "⚠️ Ollama is not available on Streamlit Cloud. Please run this locally for full LLM features."


# -------- AI Interface --------
def ai_agent_interaction(files_dict):
    st.title("AI Code Assistant 🤖")
    st.write("Welcome! Ask me questions about your repo (LLM not active on cloud).")

    menu = st.radio("Choose a task", ["Search Files", "Review Code", "Generate Test Cases", "Generate Code Summary"])

    if menu == "Search Files":
        query = st.text_input("Search the repo (e.g., 'Controller')")
        if query and st.button("🔍 Search"):
            model = SentenceTransformer('all-MiniLM-L6-v2')
            best_match = search_faiss(query, model)
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


# -------- Streamlit UI --------
st.set_page_config(page_title="AI Code Assistant", layout="wide")

model = SentenceTransformer('all-MiniLM-L6-v2')
files_dict = read_all_files(".")
if files_dict and (embedding_matrix is None or faiss_index is None):
    create_faiss_index(files_dict, model)

ai_agent_interaction(files_dict)
