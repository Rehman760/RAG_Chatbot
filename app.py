import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader,TextLoader,Docx2txtLoader,CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")

os.environ['HF_API_KEY']=os.getenv("HF_API_KEY")

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
      """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Question:{input}

    """
)

def create_vector_embeddings(uploaded_files):

     # Clear old vectors before creating new ones
    if "vectors" in st.session_state:
        del st.session_state["vectors"]
    if "final_documents" in st.session_state:
        del st.session_state["final_documents"]
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        documents = []
        for uploaded_file in uploaded_files:
            file_path = f"temp_files/{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Determine file type and use the correct loader
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif uploaded_file.name.endswith(".txt"):
                loader = TextLoader(file_path)
            elif uploaded_file.name.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif uploaded_file.name.endswith(".csv"):
                loader = CSVLoader(file_path)
            else:
                st.error(f"Unsupported file format: {uploaded_file.name}")
                continue

            documents.extend(loader.load())

        if not documents:
            st.error("No valid text extracted from the uploaded files.")
            return

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(documents)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)


st.title("RAG Document Q&A with GROQ and LLAMA 3 with Huggingface embeddings model all-MiniLM-L6-v2")

uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf", "txt", "docx", "csv"])



if st.button("Document Embedding"):
    if uploaded_files:
        create_vector_embeddings(uploaded_files)
        st.write("Vector Database is ready")
    else:
        st.write("Please upload at least one PDF file.")

user_prompt=st.text_input("Enter your query from research papers",disabled='vectors' not in st.session_state)

import time

if user_prompt:
    documents_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,documents_chain)

    start=time.process_time()
    response=retrieval_chain.invoke({'input':user_prompt})
    print(f"Response time :{time.process_time()-start}")

    st.write(response['answer'])

    ##with a streamlit expander

    with st.expander("Documents similarity Search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("----------------------")
