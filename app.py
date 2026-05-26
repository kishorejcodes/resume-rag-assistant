import os
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate

# 1. Page Configuration & Title
st.set_page_config(page_title="Kishor's RAG AI Assistant", page_icon="🤖")
st.title("📄 AI Resume Assistant")
st.write("Ask questions about Kishor's experience, skills, and projects based on his resume PDF.")

# 2. Securely get OpenAI API Key from Streamlit Secrets
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("Please configure the OPENAI_API_KEY in Streamlit Secrets.")
    st.stop()

# 3. Cache the RAG pipeline setup so it only runs ONCE when the app starts
@st.cache_resource
def initialize_rag():
    # Load PDF
    pdf_path = "KishorKumarJ_DA4_5Yrs.pdf" # Place a copy of this PDF in your project folder
    reader = PdfReader(pdf_path)
    documents = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            documents.append(Document(page_content=text))
            
    # Split, Embed, and build Vector Store
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    
    # Setup QA Chain
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    system_prompt = (
        "You are an AI assistant answering questions about Kishor's resume.\n"
        "Use the following pieces of retrieved context to answer the question.\n"
        "If you don't know the answer, say that you don't know. Keep it professional.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

# Initialize the chain
try:
    rag_chain = initialize_rag()
except Exception as e:
    st.error(f"Failed to initialize RAG pipeline: {e}")
    st.stop()

# 4. User Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_query := st.chat_input("Ask something (e.g., What is his SQL experience?)"):
    # Display human message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    # Generate and display AI response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing resume..."):
            response = rag_chain.invoke({"input": user_query})
            answer = response["answer"]
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
