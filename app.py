import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import PyPDF2
import tempfile
import os

def load_pdf_text(file):
    reader = PyPDF2.PdfReader(file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

def split_text(text, chunk_size=500, overlap=50):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]

def load_local_llm():
    model_id = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
    return HuggingFacePipeline(pipeline=pipe)

def create_qa_chain(chunks):
    docs = [Document(page_content=chunk) for chunk in chunks]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    llm = load_local_llm()
    return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# --- Streamlit UI ---
st.set_page_config(page_title="📄 Local PDF Q&A", layout="wide")
st.title("📄 Ask Questions from Your PDF (Local LLM)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Reading PDF..."):
        text = load_pdf_text(uploaded_file)
        chunks = split_text(text)

    with st.spinner("Loading embeddings and LLM..."):
        qa_chain = create_qa_chain(chunks)

    st.success("PDF processed! Ask your question below.")

    question = st.text_input("Ask a question from the document:")
    if question:
        with st.spinner("Thinking..."):
            result = qa_chain({"query": question})
            st.write("### 🧠 Answer:")
            st.info(result["result"])
