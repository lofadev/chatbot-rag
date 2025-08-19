import os
import shutil
import pickle
from typing import List, Dict, Any

import gradio as gr
from dotenv import load_dotenv

# Docling core
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker import HierarchicalChunker

# OpenAI client
from openai import OpenAI
import numpy as np

# Load API key
load_dotenv()

# Thư mục lưu file và index
DOCUMENTS_DIR = "documents"
INDEX_PATH = "vector_index.pkl"
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

# OpenAI cấu hình
_openai_client = OpenAI()
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


# ====== INDEX ======
def _load_index() -> List[Dict[str, Any]]:
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, "rb") as f:
            return pickle.load(f)
    return []


def _save_index(items: List[Dict[str, Any]]):
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(items, f)


_index: List[Dict[str, Any]] = _load_index()


def _embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    res = _openai_client.embeddings.create(input=texts, model=EMBED_MODEL)
    return [d.embedding for d in res.data]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ====== DOC CONVERT ======
def convert_with_docling(file_path: str) -> List[str]:
    """
    Đọc file bằng Docling và chunk thành danh sách đoạn văn bản.
    """
    converter = DocumentConverter()
    chunker = HierarchicalChunker()
    result = converter.convert(file_path)
    docling_doc = result.document
    chunks = list(chunker.chunk(docling_doc))
    texts: List[str] = []
    for ch in chunks:
        content = ch.text or ""
        if content.strip():
            texts.append(content)
    return texts


# ====== QUẢN LÝ FILE/INDEX ======
def upload_file(file):
    if file is None:
        return "Không có file nào được tải lên."

    filename = os.path.basename(file.name)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".pdf", ".docx", ".txt", ".md"]:
        return f"Định dạng {ext} không được hỗ trợ."

    file_path = os.path.join(DOCUMENTS_DIR, filename)
    if os.path.exists(file_path):
        return f"File {filename} đã tồn tại."

    shutil.copy(file.name, file_path)

    # Convert -> embed -> ghi vào index nội bộ
    texts = convert_with_docling(file_path)
    if not texts:
        return f"Không đọc được nội dung từ {filename}"

    embeddings = _embed_texts(texts)
    if not embeddings:
        return f"Không thể tạo embedding cho {filename}"

    for text, emb in zip(texts, embeddings):
        _index.append(
            {
                "content": text,
                "meta": {
                    "source_file": filename,
                },
                "embedding": emb,
            }
        )

    _save_index(_index)
    return f"File {filename} đã được tải lên và đánh index thành công."


def list_documents():
    return "\n".join(os.listdir(DOCUMENTS_DIR)) or "No documents."


def delete_document(filename):
    file_path = os.path.join(DOCUMENTS_DIR, filename)
    if not os.path.exists(file_path):
        return f"Không tìm thấy file: {filename}"

    os.remove(file_path)

    # Xóa khỏi index nội bộ
    global _index
    _index = [it for it in _index if it.get("meta", {}).get("source_file") != filename]
    _save_index(_index)
    return f"File {filename} đã được xóa."


# ====== HỎI ĐÁP ======
def _build_prompt(query: str, items: List[Dict[str, Any]]) -> str:
    context_blocks = []
    for i, it in enumerate(items, start=1):
        content = it.get("content", "")
        meta = it.get("meta", {})
        source = meta.get("source_file", "?")
        cit = f"[{source}]"
        context_blocks.append(f"Nguồn {i} {cit}:\n{content}")

    context_text = "\n\n".join(context_blocks)
    prompt = (
        "Bạn là một trợ lý AI tìm kiếm thông tin thông minh.\n"
        "Sử dụng chính xác thông tin từ tài liệu dưới đây để trả lời câu hỏi.\n"
        'Nếu không biết, hãy trả lời "Tôi không biết".\n'
        "Bắt buộc phải kèm nguồn trích dẫn theo định dạng: [Tên file, Trang X]\n\n"
        f"Ngữ cảnh:\n{context_text}\n\n"
        f"Câu hỏi: {query}\n\n"
        "Trả lời:"
    )
    return prompt


def ask_question(question, num_results=5):
    if not str(question or "").strip():
        return "Vui lòng nhập câu hỏi."

    if not _index:
        return "Chưa có tài liệu nào trong hệ thống."

    # Embed câu hỏi
    q_emb = _embed_texts([question])
    if not q_emb:
        return "Không tạo được embedding cho câu hỏi."
    qv = np.array(q_emb[0], dtype=float)

    # Tính cosine với tất cả documents
    scores = []
    for it in _index:
        dv = np.array(it.get("embedding") or [], dtype=float)
        scores.append(_cosine_sim(qv, dv))

    # Lấy top-k
    k = max(1, int(num_results))
    top_idx = np.argsort(scores)[-k:][::-1]
    top_items = [_index[i] for i in top_idx]
    if not top_items:
        return "Không tìm thấy thông tin trong tài liệu."

    # Gọi OpenAI sinh câu trả lời
    prompt = _build_prompt(question, top_items)
    resp = _openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "Bạn trả lời bằng tiếng Việt."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return (
        resp.choices[0].message.content
        if resp.choices
        else "Không tạo được câu trả lời."
    )


# ========== GRADIO UI ==========
with gr.Blocks(title="RAG QA - Docling + OpenAI (Local Vector Store)") as demo:
    gr.Markdown("# 🤖 RAG-based QA System (Docling + OpenAI)")

    with gr.Tab("📄 Quản lý Tài liệu"):
        with gr.Row():
            file_upload = gr.File(
                label="Upload", file_types=[".pdf", ".docx", ".txt", ".md"]
            )
            upload_btn = gr.Button("📤 Upload & Index")
            upload_out = gr.Textbox(label="Kết quả", lines=5)

        list_btn = gr.Button("📋 Liệt kê")
        list_out = gr.Textbox(label="Danh sách", lines=5)

        delete_in = gr.Textbox(label="Tên file để xóa")
        delete_btn = gr.Button("🗑️ Xóa")
        delete_out = gr.Textbox(label="Kết quả", lines=5)

        upload_btn.click(upload_file, inputs=file_upload, outputs=upload_out)
        list_btn.click(list_documents, outputs=list_out)
        delete_btn.click(delete_document, inputs=delete_in, outputs=delete_out)

    with gr.Tab("💬 Hỏi đáp"):
        chatbot = gr.Chatbot(height=400)
        question_in = gr.Textbox(label="Câu hỏi")
        submit_btn = gr.Button("🚀 Gửi")
        num_results = gr.Slider(1, 10, value=5, step=1, label="Số kết quả")

        def user(message, history):
            return "", history + [[message, None]]

        def bot(history, num_results):
            q = history[-1][0]
            answer = ask_question(q, num_results)
            history[-1][1] = answer
            return history

        question_in.submit(user, [question_in, chatbot], [question_in, chatbot]).then(
            bot, [chatbot, num_results], chatbot
        )
        submit_btn.click(user, [question_in, chatbot], [question_in, chatbot]).then(
            bot, [chatbot, num_results], chatbot
        )


demo.launch(server_name="localhost", server_port=7860)
