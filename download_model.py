from huggingface_hub import hf_hub_download

def download_model():
    print("Downloading model from HuggingFace...")
    model_path = hf_hub_download(
        repo_id="TeichAI/Qwen3-4B-Thinking-2507-MiniMax-M2.1-Distill-GGUF",
        filename="Qwen3-4B-Thinking-2507-MiniMax-M2.1-Distill.iq4_nl.gguf",
        local_dir="./models"
    )
    print(f"Model downloaded to: {model_path}")
    return model_path

if __name__ == "__main__":
    download_model()