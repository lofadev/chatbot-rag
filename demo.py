import os
import shutil
import gradio as gr
from langchain.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Tải các biến môi trường (cho OpenAI API key)
load_dotenv()

# Các thư mục
DOCUMENTS_DIR = "documents"
INDEX_DIR = "faiss_index"

# Đảm bảo các thư mục tồn tại
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Cấu hình mặc định cho chunking
DEFAULT_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "strategy": "recursive",  # "recursive", "character", "token"
    "separators": ["\n\n", "\n", " ", ""],
    "length_function": "len",
}


# Hàm lấy loader phù hợp dựa trên phần mở rộng của file
def get_loader(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PyMuPDFLoader(file_path, extract_tables="markdown", mode="page")
    elif ext == ".docx":
        return Docx2txtLoader(file_path)
    elif ext in [".txt", ".md"]:
        return TextLoader(file_path)
    else:
        raise ValueError("Unsupported file format")


# Hàm xử lý một tài liệu đơn lẻ với metadata chi tiết
def ingest_document(file_path, config=None):
    if config is None:
        config = DEFAULT_CONFIG.copy()

    filename = os.path.basename(file_path)
    loader = get_loader(file_path)
    documents = loader.load()

    # Cải thiện metadata cho mỗi document
    for doc in documents:
        doc.metadata.update(
            {
                "source_file": filename,
                "file_path": file_path,
                "file_type": os.path.splitext(filename)[1].lower(),
                "processed_time": str(os.path.getmtime(file_path)),
            }
        )

        # Thêm page number nếu có (cộng thêm 1 để bắt đầu từ trang 1)
        if "page" in doc.metadata:
            page_info = doc.metadata.get("page")
            if page_info is not None and isinstance(page_info, (int, float)):
                doc.metadata["page_number"] = int(page_info) + 1
            else:
                doc.metadata["page_number"] = page_info
        else:
            # Không có thông tin trang
            doc.metadata["page_number"] = "N/A"

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        length_function=len,
        separators=config.get("separators", ["\n\n", "\n", " ", ""]),
    )
    chunks = text_splitter.split_documents(documents)

    # Thêm thông tin chunk index cho mỗi chunk
    for i, chunk in enumerate(chunks):
        chunk.metadata.update(
            {
                "chunk_index": i,
                "chunk_size": len(chunk.page_content),
                "total_chunks": len(chunks),
            }
        )

    return chunks


# Hàm xây dựng hoặc xây dựng lại index với cấu hình chunking
def build_index(config=None):
    if config is None:
        config = DEFAULT_CONFIG.copy()

    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)

    all_chunks = []
    processed_files = []

    for filename in os.listdir(DOCUMENTS_DIR):
        if filename.startswith("."):  # Bỏ qua hidden files
            continue

        file_path = os.path.join(DOCUMENTS_DIR, filename)
        if os.path.isfile(file_path):
            try:
                chunks = ingest_document(file_path, config)
                all_chunks.extend(chunks)
                processed_files.append(f"{filename}: {len(chunks)} chunks")
            except Exception as e:
                processed_files.append(f"{filename}: Error - {str(e)}")

    if all_chunks:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = FAISS.from_documents(all_chunks, embeddings)
        vectorstore.save_local(INDEX_DIR)

        result = f"Index built successfully with {len(all_chunks)} total chunks.\n"
        result += f"Chunking strategy: {config['strategy']}, "
        result += f"Chunk size: {config['chunk_size']}, "
        result += f"Overlap: {config['chunk_overlap']}\n\n"
        result += "Processed files:\n" + "\n".join(processed_files)
        return result
    else:
        return "No documents to index."


# Hàm tải vectorstore
def load_vectorstore():
    if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        return FAISS.load_local(
            INDEX_DIR, embeddings, allow_dangerous_deserialization=True
        )
    return None


# Hàm tải lên và xử lý file
def upload_file(file):
    if file is None:
        return "No file uploaded."

    filename = os.path.basename(file.name)
    file_path = os.path.join(DOCUMENTS_DIR, filename)
    shutil.copy(file.name, file_path)

    config = DEFAULT_CONFIG.copy()
    result = build_index(config)  # Rebuild index sau khi tải lên
    return f"File {filename} uploaded successfully.\n\n{result}"


# Hàm liệt kê các tài liệu với thông tin chi tiết
def list_documents():
    files = os.listdir(DOCUMENTS_DIR)
    if not files:
        return "No documents."

    file_info = []
    for filename in files:
        if filename.startswith("."):
            continue
        file_path = os.path.join(DOCUMENTS_DIR, filename)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            size_mb = round(size / (1024 * 1024), 2)
            ext = os.path.splitext(filename)[1].lower()
            file_info.append(f"📄 {filename} ({size_mb} MB, {ext})")

    return "\n".join(file_info) if file_info else "No valid documents."


# Hàm xóa tài liệu
def delete_document(filename):
    file_path = os.path.join(DOCUMENTS_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        config = DEFAULT_CONFIG.copy()
        result = build_index(config)  # Rebuild index sau khi xóa
        return f"File {filename} deleted successfully.\n\n{result}"
    return "File not found."


# Hàm đánh index lại với cấu hình hiện tại
def reindex():
    config = DEFAULT_CONFIG.copy()
    return build_index(config)


# Prompt tùy chỉnh cho LLM với hướng dẫn trích dẫn chi tiết
prompt_template = """
Bạn là một trợ lý AI tìm kiếm thông tin thông minh, bạn có thể tìm kiếm thông tin trong các tài liệu đã được đánh index.
Sử dụng CHÍNH XÁC thông tin từ các đoạn văn bản dưới đây để trả lời câu hỏi. 
Nếu bạn không biết câu trả lời từ ngữ cảnh đã cho, hãy nói rằng bạn không biết, đừng bịa đặt thông tin.
Sử dụng ngôn ngữ tự nhiên, thân thiện với người dùng.
Trả lời bằng cùng ngôn ngữ với câu hỏi (Tiếng Việt hoặc Tiếng Anh).
Khi câu hỏi bằng tiếng Anh nhưng context bằng tiếng Việt, hãy hiểu context tiếng Việt, trích xuất thông tin liên quan, và soạn câu trả lời hoàn chỉnh bằng tiếng Anh. Ngược lại Tiếng Anh cũng vậy.

QUY TẮC TRÍCH DẪN BẮT BUỘC:
- BẮT BUỘC phải trích dẫn nguồn cho MỖI thông tin bạn cung cấp
- SỬ DỤNG CHÍNH XÁC citation đã được cung cấp trong từng đoạn văn bản
- Mỗi đoạn văn bản đã có sẵn "Nguồn trích dẫn:" ở cuối - PHẢI sử dụng đúng citation này
- KHÔNG được tự tạo citation mới, chỉ sử dụng citation có sẵn trong ngữ cảnh
- Khi tham khảo thông tin từ một đoạn văn bản, LUÔN include citation của đoạn đó
- Nếu thông tin đến từ nhiều đoạn văn bản, liệt kê TẤT CẢ citations liên quan

CÁCH TRÍCH DẪN:
- Sau mỗi thông tin, thêm citation trong ngoặc vuông
- Ví dụ: "Doanh thu năm 2023 là 100 triệu đồng [bao_cao_tai_chinh.pdf, Trang 5]"
- Với nhiều nguồn: "Thông tin này được xác nhận [file1.pdf, Trang 2] [file2.pdf, Trang 7]"

Ngữ cảnh với citation:
{context}

Câu hỏi: {question}

Trả lời chi tiết (BẮT BUỘC bao gồm trích dẫn nguồn cho mọi thông tin):
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


# Hàm cải thiện context với metadata chi tiết - chỉ hiển thị file và trang
def format_docs_with_metadata(docs):
    formatted_docs = []
    for doc in docs:
        metadata = doc.metadata
        source_file = metadata.get("source_file", "Unknown file")

        # Xử lý số trang (ưu tiên page_number đã được xử lý, không thì xử lý page gốc)
        if "page_number" in metadata and metadata["page_number"] != "N/A":
            # page_number đã được xử lý trong ingest_document
            display_page = metadata["page_number"]
            citation = f"[{source_file}, Trang {display_page}]"
        elif "page" in metadata and metadata["page"] != "N/A":
            # Fallback: sử dụng page gốc và cộng thêm 1
            page_info = metadata["page"]
            if isinstance(page_info, (int, float)):
                display_page = int(page_info) + 1
            else:
                display_page = page_info
            citation = f"[{source_file}, Trang {display_page}]"
        else:
            # Không có thông tin trang, chỉ hiển thị file
            citation = f"[{source_file}]"

        # Format content với citation rõ ràng hơn - không hiển thị thông tin chunk
        content = f"=== ĐOẠN VĂN BẢN TỪ {citation} ===\n{doc.page_content}\n=== KẾT THÚC ĐOẠN VĂN BẢN ===\nNguồn trích dẫn: {citation}"
        formatted_docs.append(content)

    return "\n\n---\n\n".join(formatted_docs)


# Hàm cho hỏi đáp với citation chính xác
def ask_question(question, num_results=5):
    vectorstore = load_vectorstore()
    if vectorstore is None:
        return "Không có index nào khả dụng. Vui lòng tải lên tài liệu trước."

    # Tạo LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Tạo retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": num_results})

    # Retrieve documents
    relevant_docs = retriever.get_relevant_documents(question)

    if not relevant_docs:
        return "Không tìm thấy thông tin liên quan trong tài liệu."

    # Format context với metadata chi tiết
    formatted_context = format_docs_with_metadata(relevant_docs)

    # Tạo prompt với context đã được format
    formatted_prompt = PROMPT.format(context=formatted_context, question=question)

    # Gọi LLM với prompt đã format
    result = llm.invoke(formatted_prompt)

    # Trả về nội dung text từ response
    if hasattr(result, "content"):
        return result.content
    else:
        return str(result)


# Giao diện Gradio
with gr.Blocks(title="RAG Document QA System", theme=gr.themes.Default()) as demo:
    gr.Markdown("# 🤖 RAG-based Document QA System")
    gr.Markdown(
        "Hệ thống hỏi đáp thông minh dựa trên tài liệu với khả năng trích dẫn nguồn chi tiết"
    )

    with gr.Tab("📄 Quản lý Tài liệu"):
        gr.Markdown("### Tải lên và quản lý tài liệu")

        with gr.Row():
            with gr.Column(scale=2):
                file_upload = gr.File(
                    label="Chọn tài liệu (PDF, DOCX, TXT, MD)",
                    file_types=[".pdf", ".docx", ".txt", ".md"],
                )
                upload_button = gr.Button("📤 Tải lên và Đánh index", variant="primary")
                upload_output = gr.Textbox(label="Trạng thái tải lên", lines=5)

            with gr.Column(scale=1):
                list_button = gr.Button("📋 Liệt kê tài liệu")
                documents_list = gr.Textbox(label="Danh sách tài liệu", lines=8)

        gr.Markdown("---")

        with gr.Row():
            delete_input = gr.Textbox(
                label="Tên file cần xóa", placeholder="Ví dụ: document.pdf"
            )
            delete_button = gr.Button("🗑️ Xóa tài liệu", variant="stop")
            delete_output = gr.Textbox(label="Trạng thái xóa", lines=3)

        with gr.Row():
            reindex_button = gr.Button("🔄 Đánh index lại toàn bộ", variant="secondary")
            reindex_output = gr.Textbox(label="Trạng thái đánh index", lines=5)

        # Event handlers cho tab Admin
        upload_button.click(upload_file, inputs=file_upload, outputs=upload_output)
        list_button.click(list_documents, outputs=documents_list)
        delete_button.click(delete_document, inputs=delete_input, outputs=delete_output)
        reindex_button.click(reindex, outputs=reindex_output)

    with gr.Tab("💬 Hỏi đáp"):
        gr.Markdown("### Đặt câu hỏi về tài liệu")
        gr.Markdown(
            "Hệ thống sẽ tìm kiếm thông tin liên quan và trả lời kèm trích dẫn nguồn"
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=500, label="Cuộc hội thoại", show_label=True
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="Câu hỏi của bạn",
                        placeholder="Ví dụ: Tài liệu nói gì về chính sách bảo mật?",
                        lines=2,
                        scale=4,
                    )
                    submit_btn = gr.Button("🚀 Gửi", variant="primary", scale=1)
                    clear_btn = gr.Button("🗑️ Xóa", variant="secondary", scale=1)

            with gr.Column(scale=1):
                gr.Markdown("### ⚡ Tùy chọn tìm kiếm")
                num_results = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Số kết quả tìm kiếm",
                    info="Nhiều kết quả = ngữ cảnh phong phú hơn nhưng chậm hơn",
                )

                gr.Markdown("### 📝 Gợi ý sử dụng")
                gr.Markdown(
                    """
                - Đặt câu hỏi rõ ràng và cụ thể
                - Sử dụng từ khóa có trong tài liệu
                - Câu trả lời sẽ kèm theo trích dẫn nguồn (ví dụ: [Tên file, Trang X])
                """
                )

        def user(user_message, history):
            return "", history + [[user_message, None]]

        def bot(history, num_results):
            if history and history[-1][0]:
                bot_message = ask_question(history[-1][0], num_results)
                history[-1][1] = bot_message
            return history

        # Event handlers cho tab hỏi đáp
        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, [chatbot, num_results], chatbot
        )
        submit_btn.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, [chatbot, num_results], chatbot
        )
        clear_btn.click(lambda: None, None, chatbot, queue=False)


demo.launch(share=False, server_name="localhost", server_port=7860)
