import os
# Ensure your OpenAI API key is set
os.environ["OPENAI_API_KEY"] = "your-actual-api-key"

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# FIXED: Updated modern chain imports
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# 1. Load the PDF directly using pypdf
reader = PdfReader(r"C:\Users\kisho\OneDrive\Desktop\KishorKumarJ_DA4_5Yrs.pdf")
documents = []
for page in reader.pages:
    text = page.extract_text()
    if text:
        documents.append(Document(page_content=text))

# 2. Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# 3. Create embeddings and save vector store
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)
db.save_local("vector_store")

# 4. Load vector DB
db = FAISS.load_local(
    "vector_store", 
    embeddings, 
    allow_dangerous_deserialization=True
)

# 5. Setup retriever and LLM
retriever = db.as_retriever(search_kwargs={"k": 5})
llm = ChatOpenAI(model="gpt-4o-mini")

# 6. Create modern QA Chain
system_prompt = (
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know.\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 7. Chat loop
while True:
    query = input("Question: ")
    if query.lower() == 'exit':
        break
    else:
        response = rag_chain.invoke({"input": query})
        print(response["answer"], "\n")
