import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()

# Creat token from hugging face website and use it.
#load hugging face token key key
os.environ['HF_TOKEN'] = ''
os.environ['HUGGINGFACEHUB_API_TOKEN'] = ''

st.markdown("<h1 style='text-align: center;'>CLEVER BOT</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>News Research Tool 📈</h3>", unsafe_allow_html=True)
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_bge_streamlit.pkl"   # TO SAVE EMBEDDINGS

main_placeholder = st.empty()
# llm = OpenAI(temperature=0.9, max_tokens=500)     # IT is a openAI api to create embedding 
# Here we are using bge model from hugging face as a  embedding model
#from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings

model_name = "BAAI/bge-base-en-v1.5"
model_kwargs = {"device":'cpu'}
encode_kwargs = {'normalize_embeddings':True}

hf = HuggingFaceEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
)

#we are using MISTRAL AI to process the retrieved documents and the query and generate human-like text.
#LLMs are specifically designed and trained for generation tasks like answering questions, summarizing text, etc.
#The LLM takes the retrieved documents and the query, and produces a coherent, human-readable response.
#Mistral AI is an open-source LLM alternative to OpenAI's models that can be used without cost.
#OpenAI's ChatGPT (or other LLMs like GPT-3, GPT-4) is not entirely free to use in our project, especially for API-based integration.
#Open-source LLMs: Other models like LLaMA, Falcon, or GPT-Neo (from EleutherAI) are available for free

from langchain.llms import HuggingFaceHub

hf_mistral = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-v0.1",
    model_kwargs={"temperature":0.1, "max_length":500})
hf_mistral.client.api_url = 'https://api-inference.huggingface.co/models/mistralai/Mistral-7B-v0.1'

prompt_template = """Answer to your Question is retrieved from Pieces of Context Extracted from Given URL.
 
 Context used to Retreive Answer:
 {context}

 Question:
 {question}

"""

PROMPT = PromptTemplate(
 template=prompt_template, input_variables=["context","question"]
)



if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    main_placeholder.text(f'No. of Chunks: {len(docs)}')
    # create embeddings and save it to FAISS index
    #embeddings = OpenAIEmbeddings()
    vectorstore_bge = FAISS.from_documents(docs, hf)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_bge, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})
            
            retrievalQA = RetrievalQA.from_chain_type(llm=hf_mistral,chain_type="stuff",retriever=retriever,
                                                      return_source_documents=True,chain_type_kwargs={"prompt": PROMPT})
            result = retrievalQA.invoke({"query": query})
            #chain = RetrievalQAWithSourcesChain.from_llm(llm=hf_mistral, retriever=retriever)
            #result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Helpful Answer")
            st.write(result["result"])
            

