import os
import shutil
import gradio as gr
from docling.document_converter import DocumentConverter
from haystack import Pipeline
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.dataclasses import Document
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack.components.writers import DocumentWriter
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from dotenv import load_dotenv

load_dotenv()

# Các thư mục
DOCUMENTS_DIR = "documents"

# Đảm bảo các thư mục tồn tại
os.makedirs(DOCUMENTS_DIR, exist_ok=True)


# Hàm xử lý một tài liệu đơn lẻ với metadata chi tiết sử dụng Docling và Haystack
def ingest_document(file_path):
    try:
        filename = os.path.basename(file_path)

        # Sử dụng DocumentConverter từ Docling
        try:
            converter = DocumentConverter()
            result = converter.convert(file_path)
            doc = result.document
        except Exception as e:
            raise Exception(f"Lỗi khi tải tài liệu {filename}: {str(e)}")

        # Kiểm tra nội dung tài liệu
        if not doc:
            raise ValueError(
                f"Tài liệu {filename} không có nội dung hoặc không thể đọc được"
            )

        ext = os.path.splitext(filename)[1].lower()
        processed_time = str(os.path.getmtime(file_path))

        # Tạo Haystack Documents với page info nếu có
        documents = []
        if doc.pages:  # Nếu có pages (thường cho PDF)
            for page_no in sorted(doc.pages.keys()):
                try:
                    content = doc.export_to_markdown(page_no=page_no)
                except Exception as e:
                    content = ""  # Nếu lỗi, bỏ qua hoặc xử lý
                meta = {
                    "source_file": filename,
                    "file_path": file_path,
                    "file_type": ext,
                    "processed_time": processed_time,
                    "page_number": page_no,  # Giả sử page_no bắt đầu từ 1
                }
                if content.strip():  # Chỉ thêm nếu có nội dung
                    documents.append(Document(content=content, meta=meta))
        else:  # Cho non-PDF như TXT, DOCX
            content = doc.export_to_markdown()
            meta = {
                "source_file": filename,
                "file_path": file_path,
                "file_type": ext,
                "processed_time": processed_time,
                "page_number": "N/A",
            }
            documents.append(Document(content=content, meta=meta))

        # Split documents
        try:
            text_splitter = DocumentSplitter(
                split_by="sentence",
                split_length=500,
                split_overlap=50,
            )
            splitter_pipe = Pipeline()
            splitter_pipe.add_component("splitter", text_splitter)
            result = splitter_pipe.run({"documents": documents})
            chunks = result["splitter"]["documents"]
        except Exception as e:
            raise Exception(f"Lỗi khi chia nhỏ tài liệu {filename}: {str(e)}")

        if not chunks:
            raise ValueError(f"Không thể tạo chunks từ tài liệu {filename}")

        # Add chunk info
        try:
            for i, chunk in enumerate(chunks):
                chunk.meta.update(
                    {
                        "chunk_index": i,
                        "chunk_size": len(chunk.content),
                        "total_chunks": len(chunks),
                    }
                )
        except Exception as e:
            raise Exception(
                f"Lỗi khi thêm metadata cho chunks của tài liệu {filename}: {str(e)}"
            )

        return chunks

    except (FileNotFoundError, PermissionError, ValueError, KeyError, TypeError) as e:
        # Re-raise specific exceptions với thông báo gốc
        raise e
    except Exception as e:
        raise Exception(f"Lỗi không xác định khi xử lý tài liệu {file_path}: {str(e)}")


# Hàm xây dựng hoặc xây dựng lại index với Qdrant và Haystack
def build_index():
    try:
        # Kiểm tra thư mục documents tồn tại
        if not os.path.exists(DOCUMENTS_DIR):
            raise FileNotFoundError(f"Thư mục documents không tồn tại: {DOCUMENTS_DIR}")

        # Tạo QdrantDocumentStore mới
        try:
            store = QdrantDocumentStore(
                url="http://localhost:6333",  # Persistent local path
                index="rag-demo",
                embedding_dim=1536,  # Dimension for text-embedding-3-small
                recreate_index=True,
            )
        except Exception as e:
            raise Exception(f"Lỗi khi tạo QdrantDocumentStore: {str(e)}")

        all_chunks = []
        processed_files = []
        error_files = []

        # Lấy danh sách files
        try:
            file_list = os.listdir(DOCUMENTS_DIR)
        except PermissionError:
            raise PermissionError(f"Không có quyền đọc thư mục: {DOCUMENTS_DIR}")
        except Exception as e:
            raise Exception(f"Lỗi khi đọc thư mục documents: {str(e)}")

        # Xử lý từng file
        for filename in file_list:
            if filename.startswith("."):  # Bỏ qua hidden files
                continue

            file_path = os.path.join(DOCUMENTS_DIR, filename)

            # Chỉ xử lý files, không phải directories
            if not os.path.isfile(file_path):
                continue

            try:
                chunks = ingest_document(file_path)
                all_chunks.extend(chunks)
                processed_files.append(f"{filename}: {len(chunks)} chunks")
            except Exception as e:
                error_message = f"{filename}: {str(e)}"
                error_files.append(error_message)
                processed_files.append(error_message)

        # Kiểm tra có documents để index không
        if not all_chunks:
            if error_files:
                error_summary = "\n".join(error_files)
                return (
                    f"Không thể tạo index - tất cả files đều gặp lỗi:\n{error_summary}"
                )
            else:
                return "Không có tài liệu hợp lệ để tạo index."

        # Tạo indexing pipeline với Haystack
        try:
            indexing_pipe = Pipeline()
            indexing_pipe.add_component(
                "embedder", OpenAIDocumentEmbedder(model="text-embedding-3-small")
            )
            indexing_pipe.add_component("writer", DocumentWriter(document_store=store))
            indexing_pipe.connect("embedder", "writer")
            indexing_pipe.run({"embedder": {"documents": all_chunks}})
        except Exception as e:
            raise Exception(f"Lỗi khi indexing documents vào Qdrant: {str(e)}")

        # Tạo kết quả thành công
        success_count = len(processed_files) - len(error_files)
        result = f"🎉 Index được tạo thành công!\n"
        result += f"📊 Tổng số chunks: {len(all_chunks)}\n"
        result += f"📁 Files xử lý thành công: {success_count}/{len(processed_files)}\n"

        if error_files:
            result += f"⚠️ {len(error_files)} file(s) gặp lỗi:\n"

        result += "📋 Chi tiết xử lý:\n" + "\n".join(processed_files)
        return result

    except (TypeError, KeyError, ValueError, FileNotFoundError, PermissionError) as e:
        return f"Lỗi khi tạo index: {str(e)}"
    except Exception as e:
        return f"Lỗi không xác định khi tạo index: {str(e)}"


# Hàm tải QdrantDocumentStore
def load_vectorstore():
    try:
        store = QdrantDocumentStore(
            url="http://localhost:6333",
            index="rag-demo",
            embedding_dim=1536,
            recreate_index=False,
        )
        return store

    except Exception as e:
        raise Exception(f"Lỗi không xác định khi tải vectorstore: {str(e)}")


# Hàm tải lên và xử lý file
def upload_file(file):
    try:
        # Kiểm tra file có được tải lên
        if file is None:
            return "Không có file nào được tải lên."

        # Kiểm tra file object có name attribute
        if not hasattr(file, "name") or not file.name:
            return "File không hợp lệ hoặc thiếu thông tin tên file."

        # Kiểm tra file có tồn tại trong temp location
        if not os.path.exists(file.name):
            return "File tạm thời không tồn tại. Vui lòng thử tải lên lại."

        filename = os.path.basename(file.name)

        # Validate filename
        if not filename or filename.startswith("."):
            return "Tên file không hợp lệ."

        # Kiểm tra định dạng file được hỗ trợ
        ext = os.path.splitext(filename)[1].lower()
        supported_extensions = [".pdf", ".docx", ".txt", ".md"]
        if ext not in supported_extensions:
            return f"Định dạng file {ext} không được hỗ trợ. Chỉ hỗ trợ: {', '.join(supported_extensions)}"

        # Kiểm tra kích thước file (giới hạn 100MB)
        file_size = os.path.getsize(file.name)
        max_size = 100 * 1024 * 1024  # 100MB
        if file_size > max_size:
            return f"File quá lớn ({file_size / (1024*1024):.1f} MB). Giới hạn tối đa là 100MB."

        file_path = os.path.join(DOCUMENTS_DIR, filename)

        # Kiểm tra file đã tồn tại
        if os.path.exists(file_path):
            return (
                f"File {filename} đã tồn tại. Vui lòng xóa file cũ hoặc đổi tên file."
            )

        try:
            # Copy file đến thư mục documents
            shutil.copy(file.name, file_path)
        except PermissionError:
            return f"Không có quyền ghi file vào thư mục {DOCUMENTS_DIR}."
        except shutil.Error as e:
            return f"Lỗi khi copy file: {str(e)}"
        except Exception as e:
            return f"Lỗi không xác định khi lưu file: {str(e)}"

        # Rebuild index sau khi tải lên
        try:
            result = build_index()
            return f"File {filename} đã được tải lên thành công.\n\n{result}"
        except Exception as e:
            # Nếu build index thất bại, xóa file đã upload để tránh inconsistency
            try:
                os.remove(file_path)
            except:
                pass
            return f"File {filename} đã được tải lên nhưng gặp lỗi khi đánh index: {str(e)}"

    except Exception as e:
        return f"Lỗi không xác định khi tải lên file: {str(e)}"


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
    try:
        # Validate filename
        if not filename or not isinstance(filename, str):
            return "Tên file không hợp lệ."

        # Sanitize filename để tránh path traversal
        filename = os.path.basename(filename.strip())
        if not filename or filename.startswith(".") or filename == "..":
            return "Tên file không hợp lệ hoặc không an toàn."

        file_path = os.path.join(DOCUMENTS_DIR, filename)

        # Đảm bảo file path nằm trong DOCUMENTS_DIR (security check)
        if not os.path.abspath(file_path).startswith(os.path.abspath(DOCUMENTS_DIR)):
            return "Đường dẫn file không hợp lệ."

        # Kiểm tra file có tồn tại
        if not os.path.exists(file_path):
            return f"Không tìm thấy file: {filename}"

        # Kiểm tra đây có phải là file không (không phải thư mục)
        if not os.path.isfile(file_path):
            return f"{filename} không phải là file hợp lệ."

        try:
            # Xóa file
            os.remove(file_path)
        except PermissionError:
            return f"Không có quyền xóa file: {filename}"
        except Exception as e:
            return f"Lỗi khi xóa file {filename}: {str(e)}"

        # Rebuild index sau khi xóa
        try:
            result = build_index()
            return f"File {filename} đã được xóa thành công.\n\n{result}"
        except Exception as e:
            return f"File {filename} đã được xóa nhưng gặp lỗi khi cập nhật index: {str(e)}"

    except Exception as e:
        return f"Lỗi không xác định khi xóa file: {str(e)}"


# Hàm đánh index lại với cấu hình hiện tại
def reindex():
    return build_index()


# Prompt tùy chỉnh cho LLM với hướng dẫn trích dẫn chi tiết (sử dụng Jinja2 cho Haystack PromptBuilder)
prompt_template = """
Bạn là một trợ lý AI tìm kiếm thông tin thông minh, bạn có thể tìm kiếm thông tin trong các tài liệu đã được đánh index.
Sử dụng CHÍNH XÁC thông tin từ các đoạn văn bản dưới đây để trả lời câu hỏi. 
Nếu bạn không biết câu trả lời từ ngữ cảnh đã cho, hãy nói rằng bạn không biết, đừng bịa đặt thông tin.
Sử dụng ngôn ngữ tự nhiên, thân thiện với người dùng.
Trả lời bằng cùng ngôn ngữ với câu hỏi (Tiếng Việt hoặc Tiếng Anh).

QUY TẮC TRÍCH DẪN BẮT BUỘC:
- BẮT BUỘC phải trích dẫn nguồn cho MỖI thông tin bạn cung cấp
- SỬ DỤNG CHÍNH XÁC trích dẫn đã được cung cấp trong từng đoạn văn bản
- Mỗi đoạn văn bản đã có sẵn "Nguồn trích dẫn:" ở cuối - PHẢI sử dụng đúng trích dẫn này
- KHÔNG được tự tạo trích dẫn mới, chỉ sử dụng trích dẫn có sẵn trong ngữ cảnh
- Khi tham khảo thông tin từ một đoạn văn bản, LUÔN include trích dẫn của đoạn đó
- Nếu thông tin đến từ nhiều đoạn văn bản, liệt kê TẤT CẢ trích dẫn liên quan

CÁCH TRÍCH DẪN:
- Sau mỗi thông tin, thêm citation trong ngoặc vuông
- Ví dụ: "Doanh thu năm 2023 là 100 triệu đồng [bao_cao_tai_chinh.pdf, Trang 5]"
- Với nhiều nguồn: "Thông tin này được xác nhận [file1.pdf, Trang 2] [file2.pdf, Trang 7]"

Ngữ cảnh với trích dẫn:
{% for doc in documents %}
=== ĐOẠN VĂN BẢN TỪ [{{ doc.meta.source_file }}, Trang {{ doc.meta.page_number }}] ===
{{ doc.content }}
=== KẾT THÚC ĐOẠN VĂN BẢN ===
Nguồn trích dẫn: [{{ doc.meta.source_file }}, Trang {{ doc.meta.page_number }}]
{% endfor %}

Câu hỏi: {{ question }}

Trả lời chi tiết (BẮT BUỘC bao gồm trích dẫn nguồn cho mọi thông tin):
"""


# Hàm cho hỏi đáp với citation chính xác sử dụng Haystack Pipeline
def ask_question(question, num_results=5):
    try:
        # Validate input parameters
        if not question or not isinstance(question, str):
            return "Câu hỏi không hợp lệ. Vui lòng nhập một câu hỏi."

        question = question.strip()
        if not question:
            return "Câu hỏi không thể để trống."

        if len(question) > 1000:
            return "Câu hỏi quá dài (tối đa 1000 ký tự). Vui lòng rút gọn câu hỏi."

        # Validate num_results
        if not isinstance(num_results, int) or num_results < 1 or num_results > 20:
            num_results = 5  # Set default value if invalid

        # Tải vectorstore
        try:
            store = load_vectorstore()
        except Exception as e:
            return f"Lỗi khi tải vectorstore: {str(e)}"

        if store is None:
            return "Không có index nào khả dụng. Vui lòng tải lên tài liệu và tạo index trước."

        # Tạo query pipeline với Haystack
        try:
            query_pipe = Pipeline()
            query_pipe.add_component(
                "embedder", OpenAITextEmbedder(model="text-embedding-3-small")
            )
            query_pipe.add_component(
                "retriever",
                QdrantEmbeddingRetriever(document_store=store, top_k=num_results),
            )
            query_pipe.add_component(
                "prompt_builder",
                PromptBuilder(
                    template=prompt_template, required_variables=["question"]
                ),
            )
            query_pipe.add_component(
                "generator",
                OpenAIGenerator(
                    model="gpt-4o-mini", generation_kwargs={"temperature": 0}
                ),
            )

            query_pipe.connect("embedder.embedding", "retriever.query_embedding")
            query_pipe.connect("retriever.documents", "prompt_builder.documents")
            query_pipe.connect("prompt_builder.prompt", "generator.prompt")

            result = query_pipe.run(
                {
                    "embedder": {"text": question},
                    "prompt_builder": {"question": question},
                }
            )

            response_content = result["generator"]["replies"][0]

            # Kiểm tra response không trống
            if not response_content or not response_content.strip():
                return "LLM không trả lời được câu hỏi. Hãy thử câu hỏi khác."

            return response_content.strip()

        except Exception as e:
            return f"Lỗi khi chạy query pipeline: {str(e)}"

    except Exception as e:
        return f"Lỗi không xác định khi xử lý câu hỏi: {str(e)}"


# Giao diện Gradio (giữ nguyên như gốc)
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
