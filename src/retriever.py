#Importing libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_community.chains.conversational_retrieval import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_mistralai import ChatMistralAI
# from langchain_classic.chains.retrieval import create_retrieval_chain
# from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
import os

# load_dotenv()

# mistral_api_key = os.getenv("MISTRAL_API_KEY")

file_path = r"C:\Users\shind\SKILL_UP\Projects_To_Upload\rag_eval_project\data\cms_medicare_policy.pdf"
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
db_dir = "chroma_store"

#Loading the pdf document and splitting it into manageable chunks
def load_and_chunk_document(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200       # overlap helps with context at boundaries
    )

    chunks = splitter.split_documents(docs)
    return chunks

#Embeddings & Vector Store: Convert text to vectors and store them
def vector_store():
  chunks = load_and_chunk_document(file_path)
  embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
  if os.path.exists(db_dir):
        return Chroma(persist_directory=db_dir, embedding_function=embeddings)
  vectorstore = Chroma.from_documents(chunks, embeddings)

  return vectorstore

def build_retriever():
    # llm = ChatMistralAI(model="mistral-small-latest",temperature=0.2,mistral_api_key=mistral_api_key)
    retriever = vector_store().as_retriever(search_kwargs={"k": 5})
    return retriever 


# Iterate through and inspect the chunks
if __name__ == "__main__":
    query ="What is inhouse patient"
    retriever = build_retriever()
    response = retriever.invoke(query)
    for i, chunk in enumerate(response):
        print(f"Chunk {i+1}: {chunk.page_content}")
        print(f"Metadata: {chunk.metadata}\n")