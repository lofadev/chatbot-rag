# 🤖 RAG-based Document QA System

Hệ thống hỏi đáp thông minh dựa trên tài liệu với khả năng trích dẫn nguồn chi tiết. Dự án cung cấp hai implementation khác nhau:

- **Haystack Implementation** (`haystack_rag.py`) - Sử dụng Haystack framework với Qdrant vector store
- **LangChain Implementation** (`langchain_rag.py`) - Sử dụng LangChain framework với FAISS vector store

## ✨ Tính năng chính

- 📄 **Hỗ trợ đa định dạng tài liệu**: PDF, DOCX, TXT, MD
- 🔍 **Tìm kiếm semantic thông minh** với embeddings
- 💬 **Giao diện chat trực quan** sử dụng Gradio
- 📚 **Trích dẫn nguồn chính xác** với thông tin trang và file
- 🔄 **Quản lý tài liệu động**: Upload, xóa, reindex
- ⚙️ **Cấu hình linh hoạt** cho chunking và tìm kiếm

## 🏗️ Kiến trúc hệ thống

### Haystack Implementation

```
Documents → Docling → Document Splitting → OpenAI Embeddings → Qdrant → Retrieval → OpenAI LLM → Response
```

### LangChain Implementation

```
Documents → LangChain Loaders → Text Splitting → OpenAI Embeddings → FAISS → Retrieval → OpenAI LLM → Response
```

## 📁 Cấu trúc dự án

```
chatbot-rag/
├── 📄 README.md              # Hướng dẫn chi tiết (file này)
├── 🐍 haystack_rag.py        # Implementation sử dụng Haystack
├── 🐍 langchain_rag.py       # Implementation sử dụng LangChain
├── 📦 requirements.txt       # Dependencies Python
├── ⚙️  env_template.txt       # Template file môi trường
├── 🐳 docker-compose.yml     # Cấu hình Docker Compose cho Qdrant
├── 🚀 setup.py              # Script setup tự động
├── 🔨 Makefile              # Commands tiện ích (Linux/macOS)
├── 📁 documents/            # Thư mục chứa tài liệu upload
├── 📁 faiss_index/          # Index FAISS (LangChain)
├── 📁 qdrant_storage/       # Dữ liệu Qdrant (Haystack)
└── 📄 .env                  # File cấu hình (tạo từ template)
```

## 📋 Yêu cầu hệ thống

- **Python**: 3.8 hoặc cao hơn
- **RAM**: Tối thiểu 4GB (khuyến nghị 8GB+)
- **Disk**: 2GB trống cho dependencies và indexes
- **Internet**: Để truy cập OpenAI API

## 🚀 Cài đặt và Thiết lập

### Cách 1: Setup tự động (Khuyến nghị)

```bash
# Clone repository (nếu từ git)
git clone <repository-url>
cd chatbot-rag

# Chạy script setup tự động
python setup.py

# Hoặc sử dụng Makefile (Linux/macOS)
make setup
```

### Cách 2: Setup thủ công

#### Bước 1: Clone repository và tạo virtual environment

```bash
# Clone repository (nếu từ git)
git clone <repository-url>
cd chatbot-rag

# Tạo virtual environment
python -m venv rag-env

# Kích hoạt virtual environment
# Trên Windows:
rag-env\Scripts\activate
# Trên macOS/Linux:
source rag-env/bin/activate
```

#### Bước 2: Cài đặt dependencies

```bash
# Cập nhật pip
pip install --upgrade pip

# Cài đặt tất cả dependencies từ file requirements.txt
pip install -r requirements.txt
```

**Hoặc cài đặt từng nhóm:**

```bash
# Dependencies chung
pip install python-dotenv openai gradio

# Cho Haystack implementation
pip install haystack-ai docling
pip install haystack-integrations[qdrant]

# Cho LangChain implementation
pip install langchain langchain-community langchain-openai langchain-text-splitters
pip install faiss-cpu PyMuPDF docx2txt
```

### Bước 3: Thiết lập OpenAI API Key

**Tự động (nếu đã chạy setup):**
File `.env` đã được tạo từ template, chỉ cần chỉnh sửa.

**Thủ công:**

```bash
# Copy từ template
cp env_template.txt .env

# Hoặc tạo file .env mới
touch .env  # macOS/Linux
# Hoặc tạo file .env bằng editor trên Windows
```

Chỉnh sửa file `.env` và thêm API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

**Cách lấy OpenAI API Key:**

1. Truy cập [OpenAI Platform](https://platform.openai.com/)
2. Đăng nhập và vào mục "API Keys"
3. Tạo key mới và copy vào file `.env`

### Bước 4: Thiết lập Qdrant (Chỉ cho Haystack implementation)

**Download qdrant**

```bash
docker pull qdrant/qdrant
```

**Chạy service**

```bash
# Chạy Qdrant server
docker run -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    qdrant/qdrant
```

### Bước 5: Tạo thư mục cần thiết

```bash
# Tạo thư mục documents
mkdir documents

# Tạo thư mục cho FAISS index (LangChain)
mkdir faiss_index
```

## 🎯 Cách sử dụng

#### Chạy Haystack Implementation

```bash
# Trong terminal khác, chạy app
python haystack_rag.py
```

#### Chạy LangChain Implementation

```bash
python langchain_rag.py
```

### Truy cập ứng dụng

Mở trình duyệt và truy cập: **http://localhost:7860**

## 📖 Hướng dẫn sử dụng giao diện

### Tab "📄 Quản lý Tài liệu"

1. **Upload tài liệu:**

   - Click "Chọn tài liệu" và chọn file (PDF, DOCX, TXT, MD)
   - Click "📤 Tải lên và Đánh index"
   - Chờ hệ thống xử lý và tạo index

2. **Quản lý tài liệu:**
   - Click "📋 Liệt kê tài liệu" để xem danh sách
   - Nhập tên file và click "🗑️ Xóa tài liệu" để xóa
   - Click "🔄 Đánh index lại toàn bộ" để rebuild index

### Tab "💬 Hỏi đáp"

1. **Đặt câu hỏi:**

   - Nhập câu hỏi vào ô text
   - Điều chỉnh "Số kết quả tìm kiếm" nếu cần
   - Click "🚀 Gửi" hoặc nhấn Enter

2. **Xem kết quả:**
   - Câu trả lời sẽ hiển thị kèm trích dẫn nguồn
   - Format trích dẫn: `[Tên_file.pdf, Trang X]`

## ⚙️ Cấu hình nâng cao

### Cấu hình Chunking (LangChain)

Chỉnh sửa `DEFAULT_CONFIG` trong `langchain_rag.py`:

```python
DEFAULT_CONFIG = {
    "chunk_size": 1000,        # Kích thước chunk
    "chunk_overlap": 200,      # Overlap giữa các chunk
    "strategy": "recursive",   # Chiến lược chia chunk
    "separators": ["\n\n", "\n", " ", ""],
    "length_function": "len",
}
```

### Cấu hình Model

Thay đổi model trong code:

```python
# Embedding model
model="text-embedding-3-small"  # Hoặc text-embedding-ada-002

# LLM model
model="gpt-4o-mini"  # Hoặc gpt-3.5-turbo, gpt-4
```

## 🔧 Troubleshooting

### Lỗi thường gặp

**1. Lỗi OpenAI API Key:**

```
Error: Incorrect API key provided
```

- Kiểm tra file `.env` có chứa đúng API key
- Đảm bảo API key còn hoạt động và có credit

**2. Lỗi Qdrant connection (Haystack):**

```
Error: Could not connect to Qdrant server
```

- Đảm bảo Qdrant server đang chạy trên port 6333
- Kiểm tra Docker container: `docker ps`

**3. Lỗi file không đọc được:**

```
Error: Could not load document
```

- Kiểm tra định dạng file được hỗ trợ
- Đảm bảo file không bị corrupt hoặc password-protected

**4. Lỗi memory:**

```
Error: Out of memory
```

- Giảm `chunk_size` trong config
- Giảm số lượng documents hoặc kích thước file
- Tăng RAM cho hệ thống

### Debug logs

Thêm debug logging vào code:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📊 Hiệu năng và Giới hạn

### Giới hạn file

- **Kích thước tối đa**: 100MB per file
- **Định dạng hỗ trợ**: PDF, DOCX, TXT, MD
- **Số lượng tài liệu**: Không giới hạn (phụ thuộc vào RAM)

### Hiệu năng

- **Haystack + Qdrant**: Tốt cho large-scale, production
- **LangChain + FAISS**: Tốt cho development, small-medium scale

### Chi phí OpenAI API

- **Embedding**: ~$0.0001 per 1K tokens
- **LLM**: ~$0.0015 per 1K tokens (gpt-4o-mini)

## 🆚 So sánh Implementation

| Tính năng        | Haystack                | LangChain            |
| ---------------- | ----------------------- | -------------------- |
| **Vector Store** | Qdrant (server)         | FAISS (local)        |
| **Setup**        | Phức tạp hơn            | Đơn giản hơn         |
| **Performance**  | Tốt hơn cho large-scale | Tốt cho small-medium |
| **Scalability**  | High                    | Medium               |
| **Dependencies** | Nhiều hơn               | Ít hơn               |

## 🔒 Bảo mật

- Không commit file `.env` lên git
- Bảo vệ OpenAI API key
- Kiểm tra input files trước khi upload
- Sử dụng HTTPS trong production

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Tạo Pull Request

## 📄 License

Dự án này được phát hành dưới MIT License. Xem file `LICENSE` để biết thêm chi tiết.

## 🆘 Hỗ trợ

Nếu gặp vấn đề:

1. Kiểm tra [Troubleshooting](#-troubleshooting)
2. Tạo issue trên GitHub
3. Liên hệ qua email

---

**Phiên bản**: 1.0.0  
**Cập nhật cuối**: $(date)  
**Tác giả**: Your Name
