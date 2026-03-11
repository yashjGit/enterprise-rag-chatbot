import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
import google.generativeai as genai

st.set_page_config(page_title="Enterprise RAG Chatbot", page_icon="🧠", layout="wide")
st.title("🧠 Enterprise RAG Chatbot")
st.markdown("Upload a private PDF. The AI will read it, search it mathematically, and generate a conversational answer based ONLY on the document.")

if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    st.sidebar.header("1. System Configuration")
    api_key = st.sidebar.text_input("Enter Gemini API Key (Local Mode):", type="password")

st.sidebar.header("2. Upload Knowledge Base")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")

@st.cache_resource
def load_retrieval_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

retrieval_model = load_retrieval_model()

if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []
if "doc_embeddings" not in st.session_state:
    st.session_state.doc_embeddings = None

if uploaded_file is not None and len(st.session_state.doc_chunks) == 0:
    with st.sidebar.status("Processing PDF Pipeline..."):
        st.write("Extracting and cleaning text...")
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
        clean_text = re.sub(r'\s+', ' ', text)
        
        st.write("Chunking data...")
        words = clean_text.split()
        chunk_size = 100
        overlap = 25
        clean_chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]
        st.session_state.doc_chunks = clean_chunks
        
        st.write("Generating neural embeddings...")
        st.session_state.doc_embeddings = retrieval_model.encode(clean_chunks)
        
    st.sidebar.success(f"✅ Processed {len(clean_chunks)} data chunks!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.doc_embeddings is not None:
    user_query = st.chat_input("Ask a question about the document...")

    if user_query:
        if not api_key:
            st.error("⚠️ Gemini API Key is missing! Add it to Streamlit Secrets or sidebar.")
            st.stop()
        
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        question_embedding = retrieval_model.encode([user_query])
        similarity_scores = cosine_similarity(question_embedding, st.session_state.doc_embeddings)
        best_match_index = np.argmax(similarity_scores[0])
        highest_score = similarity_scores[0][best_match_index]
        
        if highest_score > 0.30: 
            retrieved_context = st.session_state.doc_chunks[best_match_index]
            
            genai.configure(api_key=api_key)
            generative_model = genai.GenerativeModel('gemini-1.5-flash')
            
            strict_prompt = f"""
            You are a helpful corporate AI assistant. Answer the question using ONLY the context provided.
            Context: {retrieved_context}
            Question: {user_query}
            """
            
            llm_response = generative_model.generate_content(strict_prompt)
            final_answer = llm_response.text
            
            bot_response = f"{final_answer}\n\n---\n*🔍 RAG Confidence Score: {round(highest_score * 100, 1)}%*"
            
            with st.chat_message("assistant"):
                st.markdown(bot_response)
                with st.expander("View Source Text Chunk"):
                    st.info(f"...{retrieved_context}...")
                    
        else:
            bot_response = "I cannot find a confident answer to that in the uploaded document."
            with st.chat_message("assistant"):
                st.markdown(bot_response)
            
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
else:
    st.info("👈 Upload a PDF document to begin the AI analysis.")