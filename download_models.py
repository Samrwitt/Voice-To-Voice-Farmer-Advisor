import os
import sys

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: huggingface_hub library not found.")
    print("Please run: pip install huggingface_hub")
    sys.exit(1)

# The path structure expected by SentenceTransformers when SENTENCE_TRANSFORMERS_HOME=/models/sentence_transformers
model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
folder_name = model_id.replace("/", "_")
target_dir = os.path.join("models", "sentence_transformers", folder_name)

os.makedirs(target_dir, exist_ok=True)

print(f"--- Downloading {model_id} ---")
print(f"Target directory: {target_dir}")

try:
    snapshot_download(
        repo_id=model_id,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print("\n[SUCCESS] Model downloaded successfully.")
    print("You can now start the containers with 'sudo docker compose up -d'")
except Exception as e:
    print(f"\n[ERROR] Download failed: {e}")
