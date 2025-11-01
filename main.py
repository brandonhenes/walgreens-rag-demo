import os

# 1) LLM
from langchain_openai import ChatOpenAI

# 2) RAG building blocks
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIG ---
os.environ["OPENAI_API_KEY"] = "your-key-here"  # or set in env
DOC_PATH = "store_ops_notes.txt"
QUESTION = "How can a Walgreens store manager improve EBITDA in fy2026?"
# 1. LOAD THE DOC
loader = TextLoader(DOC_PATH)
docs = loader.load()  # -> list of Document objects

# 2. SPLIT INTO CHUNKS
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
)
chunks = splitter.split_documents(docs)

# 3. EMBED + BUILD VECTOR STORE
embeddings = OpenAIEmbeddings()
vectordb = FAISS.from_documents(chunks, embeddings)
retriever = vectordb.as_retriever(k=3)  # return top 3 most relevant chunks

# 4. RETRIEVE CONTEXT FOR OUR QUESTION
relevant_docs = retriever.invoke(QUESTION)
context = "\n\n".join([d.page_content for d in relevant_docs])

# 5. ASK THE LLM, FEEDING THE CONTEXT
llm = ChatOpenAI(model="gpt-4o")  # you can drop to gpt-4o-mini if you want
prompt = f"""You are a retail/ Walgreens operations assistant.
Use ONLY the context below to answer the question.
If the context does not contain the answer, say that.

Context:
{context}

Question: {QUESTION}
Answer:"""

resp = llm.invoke(prompt)
print(resp.content)
