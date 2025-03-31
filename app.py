import streamlit as st
import os
import time
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

# Initialize session state
if 'vectors' not in st.session_state:
    st.session_state.vectors = None
if 'final_documents' not in st.session_state:
    st.session_state.final_documents = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = None
if 'groq_api_key' not in st.session_state:
    st.session_state.groq_api_key = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "llama-3.2-90b-vision-preview"
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'initialization_error' not in st.session_state:
    st.session_state.initialization_error = None

# Create temp_files directory if it doesn't exist
if not os.path.exists("temp_files"):
    os.makedirs("temp_files")

embeddings = st.session_state.embeddings

def initialize_llm(api_key, model_name):
    try:
        return ChatGroq(groq_api_key=api_key, model_name=model_name)
    except Exception as e:
        st.session_state.initialization_error = str(e)
        return None

def create_vector_embeddings(uploaded_files):
    try:
        documents = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join("temp_files", uploaded_file.name)
            try:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

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
            except Exception as e:
                st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
                continue
            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)

        if not documents:
            st.error("No valid text extracted from the uploaded files.")
            return False

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = text_splitter.split_documents(documents)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.success("Vector Database is ready!")
        return True
    except Exception as e:
        st.error(f"Error creating vector embeddings: {str(e)}")
        return False

# Sidebar for configuration
with st.sidebar:
    st.title("Configuration")
    
    # API Key input
    groq_api_key = st.text_input("Enter your GROQ API Key", type="password", value=st.session_state.groq_api_key)
    if groq_api_key:
        if groq_api_key != st.session_state.groq_api_key:
            st.session_state.groq_api_key = groq_api_key
            os.environ["GROQ_API_KEY"] = groq_api_key
            st.session_state.llm = None  # Reset LLM when API key changes
            st.session_state.initialization_error = None
    
    # Model selection
    model_options = [
        "llama-3.2-90b-vision-preview",
        "llama-3.2-11b-vision-preview",
        "deepseek-r1-distill-llama-70b",
        "qwen-2.5-32b",
        "qwen-2.5-coder-32b",
        "mistral-saba-24b"
    ]
    selected_model = st.selectbox("Select GROQ Model", model_options, index=model_options.index(st.session_state.selected_model))
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        st.session_state.llm = None  # Reset LLM when model changes
        st.session_state.initialization_error = None
    
    # Initialize LLM with selected model
    if st.session_state.groq_api_key and not st.session_state.llm:
        st.session_state.llm = initialize_llm(st.session_state.groq_api_key, st.session_state.selected_model)
        if st.session_state.llm:
            st.success("GROQ model initialized successfully!")
        else:
            st.error(f"Error initializing GROQ model: {st.session_state.initialization_error}")
            st.stop()
    elif not st.session_state.groq_api_key:
        st.warning("Please enter your GROQ API Key to continue")
        st.stop()
    
    st.title("Document Upload")
    uploaded_files = st.file_uploader("Upload your documents", accept_multiple_files=True, type=["pdf", "txt", "docx", "csv"])
    
    if uploaded_files:
        if uploaded_files != st.session_state.uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            if create_vector_embeddings(uploaded_files):
                st.success("Documents processed successfully!")
            else:
                st.error("Failed to process documents. Please try again.")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Question:{input}
    """
)

# Main content
st.title("RAG Document Q&A with GROQ models and Huggingface embeddings model all-MiniLM-L6-v2")

if st.session_state.vectors is None:
    st.info("Please upload documents in the sidebar to start querying.")
else:
    user_prompt = st.text_input("Enter your query from research papers")

    if user_prompt:
        try:
            if not st.session_state.llm:
                st.error("LLM not initialized. Please check your API key and model selection in the sidebar.")
                st.stop()
                
            documents_chain = create_stuff_documents_chain(st.session_state.llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, documents_chain)

            with st.spinner("Processing your query..."):
                start = time.process_time()
                response = retrieval_chain.invoke({'input': user_prompt})
                print(f"Response time: {time.process_time()-start}")

                st.write(response['answer'])

                with st.expander("Documents similarity Search"):
                    for i, doc in enumerate(response['context']):
                        st.write(doc.page_content)
                        st.write("----------------------")
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            if "API key" in str(e).lower():
                st.info("Please check your GROQ API key in the sidebar.")
