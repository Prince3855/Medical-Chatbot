import os
from dotenv import load_dotenv
from flask import Flask, render_template, jsonify, request
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import CTransformers

from src.helper import generate_embeddings_model
from src.prompt import *


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX_NAME = "medical-chatbot"

# Generate embeddings for the text chunks using a pre-trained language model
embeddings = generate_embeddings_model()

# Initialize a connection to the Pinecone vector database using the provided API key
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create an index in Pinecone for storing the medical book embeddings and Loading it here
index = pc.Index(PINECONE_INDEX_NAME)

# Set up the vector store using the Pinecone index and the pre-trained embeddings model
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Prompt template for the RetrievalQA chain
PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})

# Initialize the RetrievalQA chain
retriever = vector_store.as_retriever(search_kwargs={'k': 2})
qa= RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever= retriever,
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)