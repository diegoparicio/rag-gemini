import os
import pickle
import hashlib
import numpy as np
import faiss
import streamlit as st

# from dotenv import load_dotenv
# load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai

# =========================
# 🔑 CONFIG
# =========================
EMB_PATH = "embeddings.pkl"
DOCS_PATH = "docs.pkl"
INDEX_PATH = "index.faiss"
HASH_PATH = "hash.txt"

EMBEDDING_MODEL = "models/gemini-embedding-001"
# CHAT_MODEL = "gemma-3-12b-it"
CHAT_MODEL = "gemini-flash-lite-latest"

# =========================
# 🔑 CLIENTE GEMINI
# =========================
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# =========================
# 🧠 EMBEDDINGS
# =========================
def get_embedding(text: str):
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text
    )
    return np.array(result.embeddings[0].values, dtype=np.float32)

# =========================
# 🔑 HASH DOCS (detectar cambios)
# =========================
def hash_docs(docs):
    text = "".join(docs)
    return hashlib.md5(text.encode()).hexdigest()

# =========================
# 📦 GUARDAR / CARGAR
# =========================
def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# =========================
# 🚀 CLASE RAG
# =========================
class GeminiRAG:

    def __init__(self, docs):

        # 👉 IMPORTANTE: ya vienen "chunked" desde loader
        docs = [d for d in docs if d.strip()]

        if not docs:
            raise ValueError("❌ No hay documentos válidos")

        current_hash = hash_docs(docs)

        # =========================
        # 🧠 CARGAR O GENERAR
        # =========================
        if (
            os.path.exists(EMB_PATH)
            and os.path.exists(DOCS_PATH)
            and os.path.exists(INDEX_PATH)
            and os.path.exists(HASH_PATH)
        ):
            with open(HASH_PATH) as f:
                old_hash = f.read()

            if old_hash == current_hash:
                print("✅ Cargando RAG desde disco...")

                self.embeddings = load_pickle(EMB_PATH)
                self.texts = load_pickle(DOCS_PATH)
                self.index = faiss.read_index(INDEX_PATH)

            else:
                print("♻️ Documentos cambiaron → regenerando índice...")
                self._build(docs, current_hash)

        else:
            print("⚡ Primera ejecución → creando índice...")
            self._build(docs, current_hash)

        # =========================
        # 🤖 LLM
        # =========================
        self.llm = ChatGoogleGenerativeAI(
            model=CHAT_MODEL,
            temperature=0
        )

    # =========================
    # 🏗️ BUILD INDEX
    # =========================
    def _build(self, docs, current_hash):

        self.texts = docs

        # embeddings (solo una vez)
        self.embeddings = np.array(
            [get_embedding(t) for t in docs],
            dtype=np.float32
        )

        # FAISS
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

        # guardar todo
        save_pickle(self.embeddings, EMB_PATH)
        save_pickle(self.texts, DOCS_PATH)
        faiss.write_index(self.index, INDEX_PATH)

        with open(HASH_PATH, "w") as f:
            f.write(current_hash)

        print("✅ Índice creado y guardado")

    # =========================
    # 🔍 SEARCH
    # =========================
    def search(self, query, k=3):
        q_emb = get_embedding(query)
        _, idx = self.index.search(np.array([q_emb]), k)

        return [self.texts[i] for i in idx[0]]

    # =========================
    # 💬 ASK
    # =========================
    def ask(self, question):

        context = "\n\n".join(self.search(question))

        prompt = f"""
Responde usando SOLO el contexto proporcionado.

Contexto:
{context}

Pregunta:
{question}
"""
        response = self.llm.invoke(prompt)
        return response.content