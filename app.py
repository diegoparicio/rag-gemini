import streamlit as st
from rag import GeminiRAG
from loader import load_folder

import os
os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

st.set_page_config(page_title="RAG Gemini", page_icon="💬")

st.title("💬 Chat con tus documentos")
st.write("@diegoparicio")
# -------------------------
# Cargar documentos (solo 1 vez)
# -------------------------
@st.cache_resource
def init_rag():
    docs = load_folder("data/")
    return GeminiRAG(docs)

rag = init_rag()

# -------------------------
# Estado de la conversación
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------
# Mostrar historial
# -------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
# Función para limpiar respuesta
# -------------------------
def limpiar_respuesta(resp):
    try:
        if isinstance(resp, list):
            return resp[0].get("text", "")
        if hasattr(resp, "text"):
            return resp.text
        if isinstance(resp, dict):
            return resp.get("text", "")
    except Exception as e:
        print("Error limpiando respuesta:", e)

    return "No se pudo generar respuesta"

# -------------------------
# Input usuario (chat)
# -------------------------
prompt = st.chat_input("Haz una pregunta sobre tus documentos...")

if prompt:
    # Mostrar mensaje usuario
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar respuesta
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            response = rag.ask(prompt)
            respuesta = limpiar_respuesta(response)

            st.markdown(respuesta)

    # Guardar respuesta en historial
    st.session_state.messages.append({
        "role": "assistant",
        "content": respuesta
    })

# -------------------------
# Botón limpiar chat
# -------------------------
if st.button("🧹 Limpiar chat"):
    st.session_state.messages = []
    st.rerun()