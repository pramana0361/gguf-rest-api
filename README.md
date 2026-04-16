> *This repository is provided as an example/experimental project.*

> *Feel free to copy, modify, and use any part of the code without restriction.*

# GGUF REST API Server
The FastAPI-based REST API server presented in this example uses the Qwen3 thinking model in GGUF format using llama-cpp-python as the inference backend.

## Installation
#### Create and activate 'venv' (not mendatory)
```bash
python3 -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

#### Install from requirements.txt
`pip install -r requirements.txt`

## Usage
#### Download model
Edit **repo_id** and **filename** in **download_model.py**\
Model page used as an example: https://huggingface.co/TeichAI/Qwen3-4B-Thinking-2507-MiniMax-M2.1-Distill-GGUF
```python
model_path = hf_hub_download(
        repo_id="TeichAI/Qwen3-4B-Thinking-2507-MiniMax-M2.1-Distill-GGUF",
        filename="Qwen3-4B-Thinking-2507-MiniMax-M2.1-Distill.iq4_nl.gguf",
        local_dir="./models"
    )
```
#### Start download model
`python3 download_model.py`

Check whether the model file (GGUF) has been downloaded to the **/models** folder after the model download is complete.

#### Setup environment variables
Open **.env.example** 
```bash
MODEL_PATH=./models/your-downloaded-model-file.gguf
MODEL_NAME=your-downloaded-model-name
HOST=0.0.0.0
PORT=8000
N_CTX=8192 # Model context window
N_THREADS=4
N_GPU_LAYERS=0
MAX_TOKENS=2048
```
Save the file. Then, rename it and remove **".example"** from **.env.example**

#### Start server
`python3 main.py`

## API endpoints
Check server and model status\
`GET  /health`

OpenAI-compatible chat endpoint\
`POST /v1/chat/completions`

Raw text completion endpoint\
`POST /v1/completions`

## Endpoints example
### /v1/chat/completions
#### Example request:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
    "messages": [{"role": "user", "content": "What is 3x3?"}],
    "max_tokens": 64,
    "strip_thinking": false,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "stream": false
}'
```

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
    "messages": [{"role": "user", "content": "<|system|>\nYou are an AI assistant.\n<|user|>\nWhy trees occasionally resist stillness?\n<|assistant|>\n"}],
    "stop": ["<|user|>", "<|system|>"],
    "max_tokens": 128,
    "strip_thinking": false,
    "temperature": 0.7,
        "top_p": 0.9,
    "top_k": 40,
    "stream": false
}'
```

#### Example response:
```json
{
    "id": "chatcmpl-abc123",
    "model": "qwen3-4b-thinking",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "It expands to nine."},
        "finish_reason": "stop"
    }],
    "usage": {"prompt_tokens": 16, "completion_tokens": 10, "total_tokens": 26}
}
```
---
### /v1/completions
#### Example request:
```bash
curl -X POST http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
    "prompt": "What is 18 divided by 2?",
    "max_tokens": 64,
    "temperature": 0.7,
    "top_p": 0.9,
    "stream": false
}'
```

```bash
curl -X POST http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
    "prompt": "<|system|>\nYou are an AI assistant.\n<|user|>\nWhy trees occasionally resist stillness?\n<|assistant|>\n",
    "stop": ["<|user|>", "<|system|>"],
    "max_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.9,
    "stream": false
}'
```

#### Example response:
```json
{
    "model": "qwen3-4b-thinking",
    "choices": [{"text": "It equals 9.", "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 4, "completion_tokens": 50, "total_tokens": 54}
}
```

## Model training
If you want to train a model, feel free to use my notebook.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/pramana0361/d89f16f262e2575e313dfd47f5ea8ee2/py2guf_llm_trainer.ipynb)

## Closing
This project is an experimental setup exploring a FastAPI-based REST API powered by a Qwen3 thinking model in GGUF format, using llama-cpp-python as the inference backend. It’s a work in progress, intended for testing ideas and learning, so expect changes and rough edges. Contributions, feedback, and experimentation are always welcome.
