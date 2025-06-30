import os
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
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


# -------- Ask LLM via OpenAI --------
def ask_openai_llm(file_content, user_question):
    prompt = f"""Here is the file content:

{file_content}

Now answer this:
{user_question}
"""
    try:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if available
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ OpenAI API Error: {str(e)}"


# -------- AI Agent Interaction --------
def ai_agent_interaction(files_dict):
    st.title("AI Code Assistant 🤖")
    st.write("Welcome! Ask me questions about your repo using OpenAI GPT.")

    menu = st.radio("Choose a task", ["Search Files", "Review Code", "Generate Test Cases", "Generate Code Summary"])

    if menu == "Search Files":
        query = st.text_input("Search something in the repository")
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
                "Review Code": "Analyze and provide a detailed review of this code.",
                "Generate Test Cases": "Generate JUnit test cases for this code.",
                "Generate Code Summary": "Give a high-level summary of this code.",
            }
            st.write(f"Processing `{selected_file}`...")
            output = ask_openai_llm(files_dict[selected_file], prompt_map[menu])
            st.write(output)


# -------- Streamlit UI --------
st.set_page_config(page_title="AI Code Assistant", layout="wide")

model = SentenceTransformer('all-MiniLM-L6-v2')
files_dict = read_all_files(".")
if files_dict and (embedding_matrix is None or faiss_index is None):
    create_faiss_index(files_dict, model)

ai_agent_interaction(files_dict)
