import fitz  # PyMuPDF
import os
from typing import List, Dict
from langchain.schema import Document
from datetime import datetime
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  # Thay Chroma bằng FAISS
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
)  # Embedding của Google Gemini
from dotenv import load_dotenv

load_dotenv()


class EnhancedPDFLoader:
    def __init__(self, extract_images: bool = True, image_dir: str = "./images"):
        self.extract_images = extract_images
        self.image_dir = image_dir
        os.makedirs(self.image_dir, exist_ok=True)

    def load_pdf(self, pdf_path: str) -> List[Document]:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")

        # Extract text using PyMuPDFLoader (without auto-extract images to avoid base64 bloat)
        loader = PyMuPDFLoader(
            file_path=pdf_path,
            extract_images=False,  # We'll handle images manually
            mode="page",
            extract_tables="markdown",
        )
        documents = loader.load()

        # Extract images and captions, group by page
        image_metadata_by_page = self._extract_images_with_caption(pdf_path)

        # Attach image metadata to each Document (per page)
        for doc in documents:
            page_num = doc.metadata.get("page", 1)  # PyMuPDFLoader adds page metadata
            images_on_page = image_metadata_by_page.get(page_num, [])

            # Append captions to page content for better embedding/search
            if images_on_page:
                captions_text = "\n".join(
                    [
                        f"Caption of image {i+1}: {img['caption']}"
                        for i, img in enumerate(images_on_page)
                    ]
                )
                doc.page_content += f"\n\n### Images on this page:\n{captions_text}"

            # Add image paths to metadata (list of dicts)
            doc.metadata["images"] = images_on_page

        return documents

    def _extract_images_with_caption(self, pdf_path: str) -> Dict[int, List[Dict]]:
        doc = fitz.open(pdf_path)
        image_metadata_by_page = {}

        for page_num, page in enumerate(doc, start=1):
            text_blocks = page.get_text("blocks")
            images = page.get_images(full=True)
            page_images = []

            for img_index, img in enumerate(images, start=1):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)

                # Find image position
                img_rect = None
                for block in page.get_text("dict")["blocks"]:
                    if (
                        block["type"] == 1
                        and "image" in block
                        and block["image"] == xref
                    ):
                        img_rect = fitz.Rect(block["bbox"])
                        break

                # Find nearest caption (improve: check below image, threshold distance)
                caption = ""
                if img_rect:
                    min_distance = float("inf")
                    for tb in text_blocks:
                        text_rect = fitz.Rect(tb[:4])
                        dist = (
                            text_rect.tl.y - img_rect.br.y
                        )  # Distance from image bottom to text top
                        if 0 <= dist < min_distance:
                            min_distance = dist
                            caption = tb[4].strip()

                # Save image
                pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
                timestamp_ms = int(datetime.now().timestamp() * 1000)
                image_filename = f"{pdf_filename}_page{page_num}_{timestamp_ms}.png"
                image_path = os.path.join(self.image_dir, image_filename)

                if pix.n < 5:
                    pix.save(image_path)
                else:
                    pix1 = fitz.Pixmap(fitz.csRGB, pix)
                    pix1.save(image_path)
                    pix1 = None
                pix = None

                page_images.append(
                    {
                        "image_path": image_path,
                        "caption": caption,
                    }
                )

            if page_images:
                image_metadata_by_page[page_num] = page_images

        doc.close()
        return image_metadata_by_page

    def pdf_to_markdown(self, pdf_path: str) -> str:
        documents = self.load_pdf(pdf_path)
        markdown_content = []
        for i, doc in enumerate(documents, start=1):
            markdown_content.append(f"## Trang {i}")
            if doc.page_content:
                markdown_content.append(doc.page_content)
            # Add image refs in markdown
            images = doc.metadata.get("images", [])
            for img in images:
                markdown_content.append(f"![{img['caption']}]({img['image_path']})")
        return "\n\n".join(markdown_content)


# Example RAG integration with FAISS and Google Gemini Embeddings
def build_rag(pdf_path: str, persist_dir: str = "./faiss_index"):
    loader = EnhancedPDFLoader(extract_images=True)
    docs = loader.load_pdf(pdf_path)

    # Split into chunks, preserve metadata
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Embed with Google Gemini (cần set GOOGLE_API_KEY environment variable)
    # Ví dụ: os.environ["GOOGLE_API_KEY"] = "your_api_key_here"
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )  # Model embedding của Gemini

    # Create FAISS vector store
    vector_db = FAISS.from_documents(chunks, embeddings)

    # Save locally for persistence
    vector_db.save_local(persist_dir)

    return vector_db


def load_rag(persist_dir: str = "./faiss_index"):
    # Load embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load FAISS index
    vector_db = FAISS.load_local(
        persist_dir, embeddings, allow_dangerous_deserialization=True
    )  # Cẩn thận với deserialization
    return vector_db


def query_rag(vector_db, query: str):
    results = vector_db.similarity_search(query, k=3)
    response = ""
    images_to_return = []

    for res in results:
        response += f"{res.page_content}\n"
        # Collect unique images from metadata
        images = res.metadata.get("images", [])
        images_to_return.extend([img["image_path"] for img in images])

    # Dedup images
    images_to_return = list(set(images_to_return))

    return {
        "text_response": response,
        "images": images_to_return,  # Return paths/URLs for user to display
    }


if __name__ == "__main__":

    try:
        # Build vector DB
        db = build_rag("./data/bao_cao_tai_chinh.pdf")

        # Hoặc load nếu đã có
        # db = load_rag()

        # Example query
        query = "Mô tả biểu đồ doanh thu"
        result = query_rag(db, query)
        print("Text response:", result["text_response"])
        print("Images to display:", result["images"])

        # In a web app, you could do: for img in result["images"]: display_image(img)

    except FileNotFoundError:
        print("Không tìm thấy file pdf.")
    except Exception as e:
        print(f"Lỗi: {e}")
