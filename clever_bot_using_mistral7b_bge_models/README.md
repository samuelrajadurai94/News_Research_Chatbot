**PROJECT TITLE: Retrieval-Augmented-Generation(RAG) QA Chatbot using LangChain with Hugging Face LLMs**

**CLEVER Bot: News Research QA chatbot.**
CLEVERBot is a user-friendly news research tool designed for information retrieval. Users can input article URLs and ask questions to receive relevant insights from the articles related to domains like stock market, finance, etc.

This project is built by using Langchain framework with hugging face's LLMs.
I Loaded URLs to fetch article content. Processed article content through LangChain's UnstructuredURL Loader.
splitted documents to chunks of roughly 1000 characters.
Created embedding vectors using Langchain Huggingface BGE model(Beijing Academy General Embedding).
The embeddings were stored and indexed using FAISS, enhancing retrieval speed.
FAISS (Facebook AI Similarity Search) is a library that allows developers to quickly search for embeddings of multimedia documents that are similar to each other.
The FAISS index were saved in a local file path in pickle format for later use.
Created a retriever interface using vector store to construct Q & A chain using LangChain.
To process information and answer the question, we are using hugging face's MISTRAL-7B-V0.1 LLM.
The Mistral-7B-v0.1 Large Language Model (LLM) is a pretrained generative text model with 7 billion parameters.
constructed a proper prompt for our task. Used LangChain’s RetrievalQA with the given prompt.
