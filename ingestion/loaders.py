def load_txt(file_path: str):
    documents = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()

    if text:
        documents.append({
            "text": text,
            "source": file_path,
            "page": 1
        })

    return documents

from docx import Document

def load_docx(file_path: str):
    doc = Document(file_path)
    documents = []

    for i, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        if text:
            documents.append({
                "text": text,
                "source": file_path,
                "page": i + 1  # paragraph index
            })

    return documents

import fitz  # PyMuPDF

def load_pdf(file_path: str):
    documents = []
    pdf = fitz.open(file_path)

    for page_num, page in enumerate(pdf):
        text = page.get_text().strip()
        if text:
            documents.append({
                "text": text,
                "source": file_path,
                "page": page_num + 1
            })

    pdf.close()
    return documents

import os

def load_document(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        return load_txt(file_path)
    elif ext == ".docx":
        return load_docx(file_path)
    elif ext == ".pdf":
        return load_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
