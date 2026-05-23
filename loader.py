import os
import pandas as pd
from pypdf import PdfReader

# =========================
# ✂️ CHUNKING (para PDFs)
# =========================
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


# =========================
# 📄 TXT
# =========================
def load_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f.readlines()
            if line.strip()
        ]


# =========================
# 📄 PDF
# =========================
def load_pdf(file_path):
    reader = PdfReader(file_path)

    chunks = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            chunks.extend(chunk_text(text))

    return chunks


# =========================
# 📊 CSV
# =========================
def load_csv(file_path):
    df = pd.read_csv(file_path)

    return df.apply(
        lambda row: " | ".join(
            f"{col}: {row[col]}" for col in df.columns
        ),
        axis=1
    ).tolist()


# =========================
# 📁 LOADER GENERAL
# =========================
def load_folder(folder_path):
    docs = []

    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)

        if file.endswith(".txt"):
            docs.extend(load_txt(path))

        elif file.endswith(".pdf"):
            docs.extend(load_pdf(path))

        elif file.endswith(".csv"):
            docs.extend(load_csv(path))

    # limpiar vacíos
    docs = [d.strip() for d in docs if d and d.strip()]

    if not docs:
        raise ValueError("❌ No se han cargado documentos válidos")

    return docs