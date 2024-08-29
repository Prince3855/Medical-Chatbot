# Medical-Chatbot

## Local Setup

1. Clone repository
    ```
    git clone https://github.com/Prince3855/Medical-Chatbot.git
    ```

2. move into project directory
    ```
    cd Medical-Chatbot
    ```

3. Create virtual environment
    ```
    conda create -p venv python=3.9 -y
    ```

4. Activate created virtual environment
    ```
    source activate ./venv
    ```

5. Install dependencies
    ```
    pip install -r requirements.txt
    ```

6. Download model from hugging face, create model directory and move downloaded models into model directory
    - [Llama-2-7B-Chat-GGM](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q4_0.bin?download=true)