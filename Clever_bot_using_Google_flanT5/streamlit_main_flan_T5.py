import os
from dotenv import load_dotenv

import warnings
warnings.filterwarnings('ignore')

import pickle
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.document_loaders import UnstructuredURLLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.chains import RetrievalQA,LLMChain
from langchain.docstore.document import Document
import base64

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.vectorstores import faiss
from langchain.chains.question_answering import load_qa_chain
#from langchain.chat_models.openai import ChatOpenAI
#from langchain.callbacks import get_openai_callback
from htmlTemplates_FLAN_T5 import css, get_user_template, get_bot_template
from langchain.llms import huggingface_hub


# OCR for scanned PDFs
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image

# Load Hugging Face token
#os.environ['HF_TOKEN'] = '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'
#os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

# for image icons in chat box:
def convert_image_to_base64(img_path):
    with open(img_path,'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return "data:image/jpeg;base64,"+encoded_string
    
eva2_base64 = convert_image_to_base64('eva2_cropped1.jpg')
girl1_base64 = convert_image_to_base64('girl1.png')


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-xl") #HuggingFace korean NLP Instructor Embedding model.
    vectorstore = faiss.FAISS.from_texts(
        texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore): #Google's FineTuned Language Net model

    llm = huggingface_hub.HuggingFaceHub(

        repo_id="google/flan-t5-small", model_kwargs={"temperature": 0.8, "max_length": 512})

    # llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(get_user_template(message.content,girl1_base64), unsafe_allow_html=True)
        else:
            st.write(get_bot_template(message.content,eva2_base64), unsafe_allow_html=True)

#For scaned documents or pdf:
def ocr_process(pdf_bytes):
    """Process scanned PDF using OCR."""
    images = convert_from_bytes(pdf_bytes)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img)
    return text

def main():
    load_dotenv()
    st.set_page_config(page_title="CLEVER BOT: NEWS RESEARCH TOOL",
                       page_icon=":books:")
    
    st.write(css, unsafe_allow_html=True)


    st.header("CLEVER BOT: NEWS RESEARCH TOOL :books:")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Query input
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        # Custom CSS for sidebar background color
        st.markdown(
            """
            <style>
            /* Styling the Streamlit sidebar with a green background */
            section[data-testid="stSidebar"] > div:first-child {
                background-color: #4CAF50;
                color: white;
                padding-top: 0px;  /* Remove top padding */
                margin-top: 0px;   /* Remove top margin */;
            }
            /* Ensure all sidebar contents start from the top */
            section[data-testid="stSidebar"] {
                padding: 0;        /* Remove all padding */
                margin: 0;         /* Remove all margin */
            }
            /* Styling the main page with a blue background */
            div[data-testid="stAppViewContainer"] {
                background-color: #1E90FF;  /* Light blue background */
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # --- SIDEBAR FOR UPLOADING FILE ---
        st.subheader("Upload Document")

        # Upload PDF or Word document
        uploaded_files = st.sidebar.file_uploader("Upload PDF or Word Document (Maximum 5 documents)",
                                                type=["pdf", "docx", "txt"],accept_multiple_files=True,
                                                key="file_uploader")


        process_button_enabled = st.sidebar.button("Process Documents")
        st.markdown(
                """
                <style>
                /* More specific styling for the button */
                div.stButton > button {
                    background-color: #1E90FF;
                    color: black !important;  /* Force text color to black */
                    padding: 0.5em;
                    border-radius: 5px;
                    border: none;
                    cursor: pointer;
                    font-weight: bold;
                }
                div.stButton > button:hover {
                    background-color: #104E8B;
                    color: black !important;  /* Ensure text color remains black on hover */
                }
                </style>
                """,
                unsafe_allow_html=True
            )



        if uploaded_files and len(uploaded_files) > 5:
            st.error("Error: You can only upload up to 5 documents.")
            # Red, disabled button
            st.markdown(
                """
                <style>
                /* More specific styling for the button */
                div.stButton > button {
                    background-color: #FF6347;
                    color: black !important;  /* Force text color to black */
                    padding: 0.5em;
                    border-radius: 5px;
                    border: none;
                    cursor: pointer;
                    font-weight: bold;
                }
                div.stButton > button:hover {
                    background-color: #FF6347;
                    color: black !important;  /* Ensure text color remains black on hover */
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            process_button_enabled = False


        if process_button_enabled and uploaded_files is not None and len(uploaded_files) <= 5:

            docs = []

            for uploaded_file in uploaded_files:
                # Process PDFs
                with st.spinner(f'Reading Uploaded documents..'):
                    if uploaded_file.type == "application/pdf":
                        with st.spinner(f'Processing PDF document'):
                            with open("temp_uploaded_file.pdf", "wb") as f:
                                f.write(uploaded_file.getbuffer())

                            # Load PDF content using PyPDFLoader
                            loader = PyPDFLoader("temp_uploaded_file.pdf")
                            docs.extend(loader.load())
                            os.remove("temp_uploaded_file.pdf")  # Clean up temporary file

                    # Process Word documents
                    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        st.write(f'Processing Word document')

                        with open("temp_uploaded_file.docx", "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        # Load Word content using UnstructuredWordDocumentLoader
                        loader = UnstructuredWordDocumentLoader("temp_uploaded_file.docx")
                        docs.extend(loader.load())
                        os.remove("temp_uploaded_file.docx")  # Clean up temporary file

                    # Process text files
                    elif uploaded_file.type == "text/plain":
                        st.write(f'Processing text document: {uploaded_file.name}')

                        text = uploaded_file.read().decode("utf-8")
                        docs.append(Document(page_content=text))
            docs = str(docs)
            # Split documents into chunks
            with st.spinner(f'Splitting Uploaded documents into chunks..'):
                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(docs)

            # Create embeddings and store in FAISS vector database
            with st.spinner(f'Creating Embedding Vectors and storing in Vector Database..'):
                vectorstore = get_vectorstore(chunks) # 'hf' is your embeddings model
                # create conversation chain
                
                st.markdown(f"Uploaded documents have been split into {len(chunks)} chunks.<br>"
                            "Uploaded Documents were processed and stored in the vector database.<br>"
                            "Now Ask your Questions with the Clever bot...",
                            unsafe_allow_html=True)
                
                st.session_state.conversation = get_conversation_chain(vectorstore)



        # --- SIDEBAR FOR URL PROCESSING ---
        st.subheader("Process URLs")

        urls = []
        for i in range(3):
            url = st.sidebar.text_input(f"URL {i+1}")
            urls.append(url)
        process_url_clicked = st.sidebar.button("Process URLs")

        if process_url_clicked:
            with st.spinner(f'Creating Embedding Vectors and storing in Vector Database..'):
                loader = UnstructuredURLLoader(urls=urls)
                data = loader.load()
                data =str(data)

                text_splitter = CharacterTextSplitter(separator="\n",chunk_size=1000,chunk_overlap=200,length_function = len)
                docs = text_splitter.split_text(data)

                vectorstore = get_vectorstore(docs)
                
                st.markdown(f"Uploaded URLs have been split into {len(docs)} chunks.<br>"
                            "Uploaded URLs were processed and stored in the vector database.<br>"
                            "Now Ask your Questions with the Clever bot...",
                            unsafe_allow_html=True)
                
                st.session_state.conversation = get_conversation_chain(vectorstore)        

if __name__=='__main__':
    main()
        









