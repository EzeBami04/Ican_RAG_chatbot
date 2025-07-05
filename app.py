import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
import os
import time
from dotenv import load_dotenv

# ==== SETUP ====
load_dotenv()
groq_api = os.getenv("groq_api")
llm = ChatGroq(api_key=groq_api, model="Gemma2-9b-It")

prompts = ChatPromptTemplate.from_template(
    """You are a helpful assistant that answers questions and teaches users about information provided in context.
    <Context>
    {context}
    </Context>
    Question: {input}"""
)

st.set_page_config(page_title="ICAN Student Examination Assistant", page_icon="🤖")
st.title("ICAN Student Examination Assistant 🤖")

# ==== VECTOR CREATION ====
def create_vectors_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    loader = PyPDFDirectoryLoader("./pdf")
    documents = loader.load()
    if not documents:
        st.error("No PDFs found in './pdf'. Please add PDF files.")
        return None, None
    st.write(f"📂 Loaded {len(documents)} documents.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=500)
    final_doc = splitter.split_documents(documents)
    if not final_doc:
        st.error("Failed to split documents. Check if PDFs are empty or unreadable.")
        return None, None

    st.write(f"📝 Split into {len(final_doc)} chunks for embedding.")
    vectors = FAISS.from_documents(final_doc, embeddings)
    st.success("✅ FAISS vector store created.")
    return vectors, final_doc

# ==== STATE INIT ====
if "vectors" not in st.session_state:
    st.session_state.vectors = None
    st.session_state.retrieval_chain = None

# ==== LOAD PDFs & CREATE EMBEDDINGS ====
if st.button("Load PDFs & Create Search Engine"):
    with st.spinner("Loading PDFs, creating embeddings and FAISS index..."):
        vectors, docs = create_vectors_embeddings()
        if vectors:
            st.session_state.vectors = vectors
            retriever = vectors.as_retriever()
            doc_chain = create_stuff_documents_chain(llm, prompts)
            st.session_state.retrieval_chain = create_retrieval_chain(
                retriever=retriever,
                combine_docs_chain=doc_chain
            )

# ==== QUESTION INPUT ====
question = st.text_input("Ask me anything on PSAF and CSME")

# ==== HANDLE QUERY ====
if question:
    if st.session_state.retrieval_chain is None:
        st.warning("⚠️ Please first click 'Load PDFs & Create Search Engine' to process your PDFs.")
    else:
        with st.spinner("Generating your answer..."):
            start = time.process_time()
            response = st.session_state.retrieval_chain.invoke({"input": question})
            end = time.process_time()

        st.write(f"✅ Answer generated in {end - start:.2f} seconds.")
        st.markdown("### 💬 **Answer:**")
        st.write(response["answer"])

        with st.expander("📄 Source Documents"):
            for i, doc in enumerate(response["source_documents"]):
                st.markdown(f"**Document {i+1}**")
                meta = doc.metadata
                source_name = meta.get("source", "Unknown")
                snippet = doc.page_content[:500]
                st.write(f"📂 **File:** `{source_name}`")
                st.write(snippet + ("..." if len(doc.page_content) > 500 else ""))
