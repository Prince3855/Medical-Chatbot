# Medical-Chatbot
I have created a chatbot that loads a large book, embeds it, and saves the embeddings in a Pinecone DB. The chatbot uses LangChain's retrieval QA chain and a Hugging Face pretrained model to perform question-and-answer tasks.

#### This application uses the following dependencies:

- **pypdf**: A library for reading, manipulating, and writing PDF files in Python.
- **langchain**: A framework for building applications with Large Language Models (LLMs).
- **sentence_transformers**: Provides models for sentence embedding, clustering, and semantic search.
- **langchain-text-splitters**: Splits text into manageable chunks for targeted processing.
- **pinecone-client**: Python client for Pinecone, enabling vector similarity search.
- **ctransformers**: Efficiently deploys and runs transformer models in production.
- **flask**: A lightweight web framework for building web applications and APIs.

## Local Setup

1. Clone repository
    ```bash
    git clone https://github.com/Prince3855/Medical-Chatbot.git
    ```

2. move into project directory
    ```bash
    cd Medical-Chatbot
    ```

3. Create virtual environment
    ```bash
    conda create -p venv python=3.9 -y
    ```

4. Activate created virtual environment
    ```bash
    source activate ./venv
    ```

5. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```

6. Download model from hugging face, create model directory and move downloaded models into model directory
    - [Llama-2-7B-Chat-GGM](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q4_0.bin?download=true)

7. Create `.env` file using `.example.env` file and update values of variables

8. If it's first time create Index on pincone data base and to embedd given book execute command
    ```bash
    python store_index.py
    ```

9. Run application
    ```bash
    python app.py
    ```

## Demo
