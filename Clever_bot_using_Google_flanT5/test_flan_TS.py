'''from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.vectorstores import faiss
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models.openai import ChatOpenAI
from langchain.callbacks import get_openai_callback
from htmlTemplates import css, user_template, bot_template
from langchain.llms import huggingface_hub
'''
import base64

#hf = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")

def convert_image_to_base64(img_path):
    with open(img_path,'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return "data:image/jpeg;base64,"+encoded_string
    
eva2_base64 = convert_image_to_base64('eva2_cropped1.jpg')
girl1_base64 = convert_image_to_base64('girl1.png')

print(girl1_base64)