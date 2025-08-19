import os
import shutil
import gradio as gr
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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
    try:
        # Kiểm tra file tồn tại
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

        # Kiểm tra file có thể đọc được
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"Không có quyền đọc file: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return PyMuPDFLoader(file_path, extract_tables="markdown", mode="page")
        elif ext == ".docx":
            return Docx2txtLoader(file_path)
        elif ext in [".txt", ".md"]:
            return TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError(
                f"Định dạng file không được hỗ trợ: {ext}. Chỉ hỗ trợ PDF, DOCX, TXT, MD"
            )

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Lỗi file không tồn tại: {str(e)}")
    except PermissionError as e:
        raise PermissionError(f"Lỗi quyền truy cập: {str(e)}")
    except Exception as e:
        raise Exception(f"Lỗi khi tạo loader cho file {file_path}: {str(e)}")


# Hàm xử lý một tài liệu đơn lẻ với metadata chi tiết
def ingest_document(file_path, config=None):
    try:
        if config is None:
            config = DEFAULT_CONFIG.copy()

        # Validate config
        if not isinstance(config, dict):
            raise TypeError("Config phải là dictionary")

        required_keys = ["chunk_size", "chunk_overlap"]
        for key in required_keys:
            if key not in config:
                raise KeyError(f"Config thiếu key bắt buộc: {key}")
            if not isinstance(config[key], int) or config[key] <= 0:
                raise ValueError(f"Config {key} phải là số nguyên dương")

        filename = os.path.basename(file_path)

        # Sử dụng get_loader đã được cải thiện với exception handling
        try:
            loader = get_loader(file_path)
            documents = loader.load()
        except Exception as e:
            raise Exception(f"Lỗi khi tải tài liệu {filename}: {str(e)}")

        # Kiểm tra nội dung tài liệu
        if not documents:
            raise ValueError(
                f"Tài liệu {filename} không có nội dung hoặc không thể đọc được"
            )

        # Cải thiện metadata cho mỗi document
        try:
            for doc in documents:
                if not hasattr(doc, "metadata"):
                    doc.metadata = {}

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
        except Exception as e:
            raise Exception(f"Lỗi khi xử lý metadata cho tài liệu {filename}: {str(e)}")

        # Xử lý text splitting
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config["chunk_size"],
                chunk_overlap=config["chunk_overlap"],
                length_function=len,
                separators=config.get("separators", ["\n\n", "\n", " ", ""]),
            )
            chunks = text_splitter.split_documents(documents)
        except Exception as e:
            raise Exception(f"Lỗi khi chia nhỏ tài liệu {filename}: {str(e)}")

        # Kiểm tra kết quả splitting
        if not chunks:
            raise ValueError(f"Không thể tạo chunks từ tài liệu {filename}")

        # Thêm thông tin chunk index cho mỗi chunk
        try:
            for i, chunk in enumerate(chunks):
                if not hasattr(chunk, "metadata"):
                    chunk.metadata = {}

                chunk.metadata.update(
                    {
                        "chunk_index": i,
                        "chunk_size": len(chunk.page_content),
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


# Hàm xây dựng hoặc xây dựng lại index với cấu hình chunking
def build_index(config=None):
    try:
        # Validate và setup config
        if config is None:
            config = DEFAULT_CONFIG.copy()

        if not isinstance(config, dict):
            raise TypeError("Config phải là dictionary")

        # Validate required config keys
        required_keys = ["chunk_size", "chunk_overlap"]
        for key in required_keys:
            if key not in config:
                raise KeyError(f"Config thiếu key bắt buộc: {key}")
            if not isinstance(config[key], int) or config[key] <= 0:
                raise ValueError(f"Config {key} phải là số nguyên dương")

        # Kiểm tra thư mục documents tồn tại
        if not os.path.exists(DOCUMENTS_DIR):
            raise FileNotFoundError(f"Thư mục documents không tồn tại: {DOCUMENTS_DIR}")

        # Xóa index cũ nếu có
        try:
            if os.path.exists(INDEX_DIR):
                shutil.rmtree(INDEX_DIR)
        except PermissionError:
            raise PermissionError(f"Không có quyền xóa thư mục index: {INDEX_DIR}")
        except Exception as e:
            raise Exception(f"Lỗi khi xóa index cũ: {str(e)}")

        # Tạo lại thư mục index
        try:
            os.makedirs(INDEX_DIR, exist_ok=True)
        except Exception as e:
            raise Exception(f"Lỗi khi tạo thư mục index: {str(e)}")

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
                chunks = ingest_document(file_path, config)
                all_chunks.extend(chunks)
                processed_files.append(f"✅ {filename}: {len(chunks)} chunks")
            except Exception as e:
                error_message = f"❌ {filename}: {str(e)}"
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

        # Tạo embeddings và vectorstore
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        except Exception as e:
            raise Exception(
                f"Lỗi khi tạo OpenAI embeddings: {str(e)}. Kiểm tra API key và kết nối mạng."
            )

        try:
            vectorstore = FAISS.from_documents(all_chunks, embeddings)
        except Exception as e:
            raise Exception(f"Lỗi khi tạo FAISS vectorstore: {str(e)}")

        try:
            vectorstore.save_local(INDEX_DIR)
        except PermissionError:
            raise PermissionError(f"Không có quyền ghi vào thư mục index: {INDEX_DIR}")
        except Exception as e:
            raise Exception(f"Lỗi khi lưu vectorstore: {str(e)}")

        # Tạo kết quả thành công
        success_count = len(processed_files) - len(error_files)
        result = f"🎉 Index được tạo thành công!\n"
        result += f"📊 Tổng số chunks: {len(all_chunks)}\n"
        result += f"📁 Files xử lý thành công: {success_count}/{len(processed_files)}\n"
        result += f"⚙️ Cấu hình chunking: Kích thước {config['chunk_size']}, Overlap {config['chunk_overlap']}\n\n"

        if error_files:
            result += f"⚠️ {len(error_files)} file(s) gặp lỗi:\n"

        result += "📋 Chi tiết xử lý:\n" + "\n".join(processed_files)
        return result

    except (TypeError, KeyError, ValueError, FileNotFoundError, PermissionError) as e:
        return f"Lỗi khi tạo index: {str(e)}"
    except Exception as e:
        return f"Lỗi không xác định khi tạo index: {str(e)}"


# Hàm tải vectorstore
def load_vectorstore():
    try:
        # Kiểm tra thư mục index tồn tại
        if not os.path.exists(INDEX_DIR):
            raise FileNotFoundError(f"Thư mục index không tồn tại: {INDEX_DIR}")

        # Kiểm tra file index.faiss tồn tại
        index_file = os.path.join(INDEX_DIR, "index.faiss")
        if not os.path.exists(index_file):
            return None  # Không có index, trả về None (không phải lỗi)

        # Kiểm tra file index.pkl tồn tại
        pkl_file = os.path.join(INDEX_DIR, "index.pkl")
        if not os.path.exists(pkl_file):
            raise FileNotFoundError(f"File index.pkl không tồn tại: {pkl_file}")

        try:
            # Tạo embeddings
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        except Exception as e:
            raise Exception(
                f"Lỗi khi tạo OpenAI embeddings: {str(e)}. Kiểm tra API key và kết nối mạng."
            )

        try:
            # Tải vectorstore
            vectorstore = FAISS.load_local(
                INDEX_DIR, embeddings, allow_dangerous_deserialization=True
            )
            return vectorstore
        except Exception as e:
            raise Exception(f"Lỗi khi tải vectorstore từ {INDEX_DIR}: {str(e)}")

    except FileNotFoundError as e:
        raise FileNotFoundError(str(e))
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
            config = DEFAULT_CONFIG.copy()
            result = build_index(config)
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
            config = DEFAULT_CONFIG.copy()
            result = build_index(config)
            return f"File {filename} đã được xóa thành công.\n\n{result}"
        except Exception as e:
            return f"File {filename} đã được xóa nhưng gặp lỗi khi cập nhật index: {str(e)}"

    except Exception as e:
        return f"Lỗi không xác định khi xóa file: {str(e)}"


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
            vectorstore = load_vectorstore()
        except Exception as e:
            return f"Lỗi khi tải vectorstore: {str(e)}"

        if vectorstore is None:
            return "Không có index nào khả dụng. Vui lòng tải lên tài liệu và tạo index trước."

        # Tạo LLM
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        except Exception as e:
            return (
                f"Lỗi khi tạo ChatOpenAI: {str(e)}. Kiểm tra API key và kết nối mạng."
            )

        # Tạo retriever và search documents
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": num_results})
            relevant_docs = retriever.get_relevant_documents(question)
        except Exception as e:
            return f"Lỗi khi tìm kiếm tài liệu liên quan: {str(e)}"

        # Kiểm tra có tài liệu liên quan không
        if not relevant_docs:
            return "Không tìm thấy thông tin liên quan trong tài liệu. Hãy thử câu hỏi khác hoặc kiểm tra lại từ khóa."

        # Format context với metadata chi tiết
        try:
            formatted_context = format_docs_with_metadata(relevant_docs)
        except Exception as e:
            return f"Lỗi khi format context: {str(e)}"

        # Tạo prompt với context đã được format
        try:
            formatted_prompt = PROMPT.format(
                context=formatted_context, question=question
            )
        except Exception as e:
            return f"Lỗi khi tạo prompt: {str(e)}"

        # Gọi LLM với prompt đã format
        try:
            result = llm.invoke(formatted_prompt)
        except Exception as e:
            return f"Lỗi khi gọi OpenAI API: {str(e)}. Kiểm tra API key, quota và kết nối mạng."

        # Trả về nội dung text từ response
        try:
            if hasattr(result, "content"):
                response_content = result.content
            else:
                response_content = str(result)

            # Kiểm tra response không trống
            if not response_content or not response_content.strip():
                return "LLM không trả lời được câu hỏi. Hãy thử câu hỏi khác."

            return response_content.strip()

        except Exception as e:
            return f"Lỗi khi xử lý response từ LLM: {str(e)}"

    except Exception as e:
        return f"Lỗi không xác định khi xử lý câu hỏi: {str(e)}"


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
