import os
import logging
from uuid import uuid4
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf, split_text_into_chunks, generate_embeddings_model


logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX_NAME = "medical-assistant"

logging.info(f"Loaded environment variables")

# Load the medical book data from a PDF file located in the "../data/" directory
extracted_data = load_pdf("data/")
logging.info(f"Loaded pdf data")

# Split the loaded text into smaller chunks for efficient processing and storage in the vector store
text_chunks = split_text_into_chunks(extracted_data)
logging.info(f"Splited text into chunks")

# Generate embeddings for the text chunks using a pre-trained language model
logging.info(f"Downloading embedding model from huggingface...")
embeddings = generate_embeddings_model()
logging.info(f"Embrdding model downloaded")

logging.info(f"Initialize a connection to the Pinecone vector database and index...")

# Initialize a connection to the Pinecone vector database using the provided API key
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create an index in Pinecone for storing the medical book embeddings and Loading it here
index = pc.Index(PINECONE_INDEX_NAME)

# Set up the vector store using the Pinecone index and the pre-trained embeddings model
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

logging.info(f"Initialized vector store")

logging.info(f"Embedding text chunks and saving them into pinecone database. \n Process might take a time  please wait it to complete...")
# Embed current loaded PDF and save embeddings in vector store
uuids = [str(uuid4()) for _ in range(len(text_chunks))]
print("uuids done\n\n")
vector_store.add_documents(documents=text_chunks, ids=uuids)

logging.info(f"vectorization done.\n")