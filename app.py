
import streamlit as st
from dotenv import load_dotenv
import os

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from utils.pdf_reader import get_pdf_text
from utils.vector_store import get_text_chunks, create_vector_store
from utils.qa_chain import get_conversational_chain

# -----------------------
# Setup
# -----------------------

load_dotenv()

if "messages" not in st.session_state:
    st.session_state.messages = []

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📄",
    layout="wide"
)

# -----------------------
# Header
# -----------------------

st.title("📄 DocuMind AI")

st.markdown("""
### AI-Powered Document Intelligence System

Upload one or more PDFs and ask questions in natural language.

**Features**
- PDF Processing
- Semantic Search
- Gemini AI Answers
- Vector Database (FAISS)
""")

# -----------------------
# Sidebar
# -----------------------

with st.sidebar:

    st.header("📂 Upload Documents")

    pdf_docs = st.file_uploader(
        "Choose PDF files",
        accept_multiple_files=True
    )

    st.markdown("---")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.info(
        "Upload PDFs and click Process Documents."
    )

# -----------------------
# Process Documents
# -----------------------

if st.button("🚀 Process Documents"):

    if pdf_docs:

        with st.spinner("Processing documents..."):

            text = get_pdf_text(pdf_docs)

            chunks = get_text_chunks(text)

            create_vector_store(chunks)

        st.success(
            f"""
            ✅ Processing Complete

            Documents Indexed Successfully

            Chunks Created: {len(chunks)}
            """
        )

    else:
        st.warning("Please upload a PDF first.")

st.divider()

# -----------------------
# Display Chat History
# -----------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------
# Chat Input
# -----------------------

question = st.chat_input(
    "Ask a question about your documents..."
)

# -----------------------
# Answer Question
# -----------------------

if question:

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    try:

        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        db = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )

        docs = db.similarity_search(
            question,
            k=3
        )

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        model = get_conversational_chain()

        final_prompt = f"""
Answer the question using only the provided context.

If the answer is not present, say:
'Answer not found in the uploaded document.'

Context:
{context}

Question:
{question}
"""

        response = model.invoke(final_prompt)

        answer = response.content

        # Show assistant response immediately
        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        with st.expander("📚 Retrieved Context"):

            for i, doc in enumerate(docs, start=1):

                st.write(f"Chunk {i}")
                st.write(doc.page_content[:500])
                st.divider()

    except Exception as e:

        st.error(f"Error: {str(e)}")

