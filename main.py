"""
GGUF API Server
=====================
Author: pramana0361
"""

import os
import json
import re
import uvicorn
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from llama_cpp import Llama
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Read from .env with fallback defaults
MODEL_PATH    = os.getenv("MODEL_PATH", "./models/default.gguf")
MODEL_NAME    = os.getenv("MODEL_NAME", "GGUF-Model")
HOST          = os.getenv("HOST", "0.0.0.0")
PORT          = int(os.getenv("PORT", "8000"))
N_CTX         = int(os.getenv("N_CTX", "8192"))
N_THREADS     = int(os.getenv("N_THREADS", "4"))
N_GPU_LAYERS  = int(os.getenv("N_GPU_LAYERS", "0"))
MAX_TOKENS    = int(os.getenv("MAX_TOKENS", "2048"))

# ── App Instance ──────────────────────────────────────────────────────────────
# Initialize FastAPI app with title and version metadata.
# These are visible in the auto-generated docs at /docs
app = FastAPI(title="GGUF API", version="1.0.0")

# ── Default System Prompt ─────────────────────────────────────────────────────
# This prompt is automatically injected at the start of every conversation
# if the user does not provide their own system message.
# It instructs the model to:
#   - Act as a helpful assistant
#   - Respond in the same language as the user
#   - Be concise and clear
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "You MUST always respond in the language used by the user. "
    "Do NOT use any other language. "
    "Be concise and clear."
)

# ── Global Model Variable ─────────────────────────────────────────────────────
# Holds the loaded Llama model instance.
# Initialized as None and loaded during app startup.
llm = None


# ── Helper Functions ──────────────────────────────────────────────────────────
# Strip thinking text for thinking model
def strip_think(text: str) -> str:
    """
    Remove <think>...</think> blocks from model output.

    Qwen3 thinking models produce internal reasoning wrapped in <think> tags
    before giving the final answer. This function strips those blocks so only
    the clean final answer is returned to the user.

    Args:
        text (str): Raw model output that may contain <think>...</think> blocks.

    Returns:
        str: Cleaned text with all <think> blocks removed and whitespace stripped.

    Example:
        Input:  "<think>Let me think...</think>\\n\\nThe answer is 42."
        Output: "The answer is 42."
    """
    subStr = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    __, ___, after = subStr.partition("</think>")
    return after.strip()


# ── Startup Event ─────────────────────────────────────────────────────────────

@app.on_event("startup")
def load_model():
    """
    Load the GGUF model into memory when the FastAPI server starts.

    This function runs automatically once when the server boots up.
    It initializes the global `llm` variable with a Llama instance.

    Model Parameters:
        model_path  : Path to the .gguf model file
        n_ctx       : Context window size in tokens (262144 = 256K tokens)
                      Large context allows very long conversations.
                      Note: Uses more RAM with larger values.
        n_threads   : Number of CPU threads for inference (set to match your CPU cores)
        n_gpu_layers: Number of model layers to offload to GPU.
                      0 = CPU only. Set to -1 to offload all layers to GPU.
        verbose     : If False, suppresses llama.cpp debug logs

    Raises:
        RuntimeError: If the model file does not exist at MODEL_PATH.
    """
    global llm
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    print("Loading model...")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False,
    )
    print("Model loaded!")


# ── Request / Response Schemas ────────────────────────────────────────────────

class Message(BaseModel):
    """
    Represents a single message in a conversation.

    Attributes:
        role    : The speaker of the message. One of: "system", "user", "assistant"
        content : The text content of the message.

    Example:
        {"role": "user", "content": "What is the capital of France?"}
    """
    role: str
    content: str


class ChatRequest(BaseModel):
    """
    Request schema for the /v1/chat/completions endpoint.

    Attributes:
        messages       : List of conversation messages in order (system, user, assistant, ...)
        max_tokens     : Maximum number of tokens to generate in the response. Default: 2048
        temperature    : Controls randomness of output.
                         0.0 = deterministic, 1.0 = very random. Default: 0.7
        top_p          : Nucleus sampling threshold.
                         Only tokens with cumulative probability <= top_p are considered. Default: 0.9
        top_k          : Limits token selection to top K most likely tokens. Default: 40
        repeat_penalty : Penalizes repeating tokens to reduce repetition. Default: 1.1
        stream         : If True, response is streamed token by token via SSE. Default: False
        strip_thinking : If True, removes <think>...</think> blocks from response. Default: True

    Example:
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain gravity."}
            ],
            "max_tokens": 512,
            "temperature": 0.7,
            "stream": false,
            "strip_thinking": true
        }
    """
    messages: List[Message]
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 40
    repeat_penalty: Optional[float] = 1.1
    stream: Optional[bool] = False
    strip_thinking: Optional[bool] = True


class CompletionRequest(BaseModel):
    """
    Request schema for the /v1/completions endpoint.

    Unlike ChatRequest, this takes a raw text prompt instead of
    a structured list of messages. Useful for text continuation tasks.

    Attributes:
        prompt      : The raw input text to complete.
        max_tokens  : Maximum number of tokens to generate. Default: 2048
        temperature : Controls randomness. 0.0 = deterministic. Default: 0.7
        top_p       : Nucleus sampling threshold. Default: 0.9
        stream      : If True, streams response token by token via SSE. Default: False

    Example:
        {
            "prompt": "The capital of France is",
            "max_tokens": 100,
            "temperature": 0.5
        }
    """
    prompt: str
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """
    Health check endpoint.

    Returns the current server status and whether the model is loaded.
    Use this to verify the server is running before sending requests.

    Returns:
        dict: {
            "status": "ok",
            "model_loaded": true/false
        }

    Example:
        curl http://localhost:8000/health
    """
    return {"status": "ok", "model_loaded": llm is not None}


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest):
    """
    OpenAI-compatible chat completions endpoint.

    Accepts a list of messages (conversation history) and returns
    the model's next response. Compatible with OpenAI chat format.

    Behavior:
        - If no system message is provided, DEFAULT_SYSTEM_PROMPT is injected automatically.
        - If stream=True, response is sent as Server-Sent Events (SSE) token by token.
        - If stream=False, full response is returned as a single JSON object.
        - If strip_thinking=True, <think>...</think> blocks are removed from the output.

    Args:
        request (ChatRequest): The chat request payload.

    Returns:
        StreamingResponse : If stream=True, returns SSE stream of token chunks.
        dict              : If stream=False, returns full completion JSON.

    Raises:
        HTTPException 503: If the model has not been loaded yet.

    Example Request:
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

    Example Response:
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
    """
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Always inject default system prompt if none exists
    has_system = any(m["role"] == "system" for m in messages)
    if not has_system:
        messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})

    if request.stream:
        def generate():
            """
            Inner generator function for streaming responses.

            Calls llama.cpp with stream=True and yields each token chunk
            as a Server-Sent Event (SSE) formatted string.

            Yields:
                str: SSE-formatted JSON string for each token chunk.
                     Format: "data: {json}\\n\\n"
            """
            stream_output = llm.create_chat_completion(
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repeat_penalty=request.repeat_penalty,
                stream=True,
            )
            for chunk in stream_output:
                # Extract the delta content from each streamed chunk
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta:
                    data = {"choices": [{"delta": {"content": delta["content"]}, "finish_reason": None}]}
                    # Yield as SSE formatted event
                    yield f"data: {json.dumps(data)}\n\n"

        # Return a streaming HTTP response with SSE media type
        return StreamingResponse(generate(), media_type="text/event-stream")

    # Non-streaming: run full inference and return complete response
    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repeat_penalty=request.repeat_penalty,
        stream=False,
    )

    # Strip <think> block from response
    # Removes internal chain-of-thought reasoning before returning to user
    if request.strip_thinking:
        for choice in output["choices"]:
            if "message" in choice and "content" in choice["message"]:
                choice["message"]["content"] = strip_think(choice["message"]["content"])

    return {
        "id": output["id"],
        "model": MODEL_NAME,
        "choices": output["choices"],
        "usage": output.get("usage", {}),
    }


@app.post("/v1/completions")
def completions(request: CompletionRequest):
    """
    Raw text completion endpoint.

    Takes a plain text prompt and returns a text continuation.
    Unlike /v1/chat/completions, this does not use a chat format
    or inject system prompts. Best for direct text continuation tasks.

    Behavior:
        - If stream=True, response is sent as Server-Sent Events (SSE).
        - If stream=False, full response is returned as JSON.

    Args:
        request (CompletionRequest): The completion request payload.

    Returns:
        StreamingResponse : If stream=True, returns SSE stream.
        dict              : If stream=False, returns full completion JSON.

    Raises:
        HTTPException 503: If the model has not been loaded yet.

    Example Request:
        curl -X POST http://localhost:8000/v1/completions \
            -H "Content-Type: application/json" \
            -d '{
                "prompt": "What is 18 divided by 2?",
                "max_tokens": 64,
                "temperature": 0.7,
                "top_p": 0.9,
                "stream": false
            }'

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

    Example Response:
        {
            "model": "qwen3-4b-thinking",
            "choices": [{"text": "It equals 9.", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 50, "total_tokens": 54}
        }
    """
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if request.stream:
        def generate():
            """
            Inner generator function for streaming text completions.

            Yields:
                str: SSE-formatted JSON string for each token chunk.
                     Format: "data: {json}\\n\\n"
            """
            stream_output = llm(
                request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stream=True,
            )
            for chunk in stream_output:
                # Extract text token from each chunk
                text = chunk["choices"][0].get("text", "")
                if text:
                    data = {"choices": [{"text": text, "finish_reason": None}]}
                    # Yield as SSE formatted event
                    yield f"data: {json.dumps(data)}\n\n"

        # Return a streaming HTTP response with SSE media type
        return StreamingResponse(generate(), media_type="text/event-stream")

    # Non-streaming: run full inference and return complete response
    output = llm(
        request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stream=False,
    )

    return {
        "model": MODEL_NAME,
        "choices": output["choices"],
        "usage": output.get("usage", {}),
    }


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Start the FastAPI server using uvicorn.

    Configuration:
        host    : "0.0.0.0" allows connections from any network interface.
                  Use "127.0.0.1" to restrict to localhost only.
        port    : 8000 is the default port.
        reload  : False in production. Set True during development for auto-reload on file changes.

    Access:
        API      : http://localhost:8000
        Docs     : http://localhost:8000/docs
        Redoc    : http://localhost:8000/redoc
    """
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)