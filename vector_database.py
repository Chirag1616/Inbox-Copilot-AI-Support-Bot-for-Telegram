import os
import pdfplumber
import pytesseract
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Paths and config
PDFS_DIRECTORY = 'pdfs/'
FAISS_DB_PATH = 'vectorstore/db_faiss'
OLLAMA_MODEL_NAME = "deepseek-r1:7b"

def ocr_pdf(filepath):
    """Extract text from each page of PDF using pdfplumber + pytesseract OCR."""
    texts = []
    try:
        with pdfplumber.open(filepath) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # convert page to image
                pil_image = page.to_image(resolution=300).original
                # OCR text extraction
                text = pytesseract.image_to_string(pil_image)
                if text.strip():
                    texts.append((page_num, text))
    except Exception as e:
        print(f"❌ Failed to OCR {filepath}: {e}")
    return texts

def load_all_pdfs(data_path):
    all_docs = []
    for filename in os.listdir(data_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(data_path, filename)
            texts = ocr_pdf(filepath)
            for page_num, page_text in texts:
                all_docs.append(Document(page_content=page_text, metadata={"source": filename, "page": page_num}))
    return all_docs


documents = load_all_pdfs(PDFS_DIRECTORY)

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    return chunks
text_chunks = create_chunks(documents)
def get_embedding_model(model_name):
    embeddings = OllamaEmbeddings(model=model_name)
    return embeddings
'''
if __name__ == "__main__":
    print("🔍 Loading PDFs from:", PDFS_DIRECTORY)
    documents = load_all_pdfs(PDFS_DIRECTORY)
    print(f"✅ Loaded {len(documents)} documents.")

    if not documents:
        raise ValueError("No documents loaded. Check your PDFs or OCR setup.")

    print("🧩 Splitting documents into chunks...")
    chunks = create_chunks(documents)
    print(f"✅ Created {len(chunks)} text chunks.")

    if not chunks:
        raise ValueError("No text chunks created. Cannot proceed.")

    print("🧠 Generating embeddings with Ollama...")
    embeddings = get_embedding_model(OLLAMA_MODEL_NAME)

    print("💾 Saving vectors to FAISS database...")
    print(f"✅ Vector DB saved to {FAISS_DB_PATH}")
'''
FAISS_DB_PATH="vectorstore/db_faiss"
faiss_db=FAISS.from_documents(text_chunks, get_embedding_model(OLLAMA_MODEL_NAME))
faiss_db.save_local(FAISS_DB_PATH)
    
    
