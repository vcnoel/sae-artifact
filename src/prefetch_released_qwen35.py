# Prefetch for released_qwen35.py, network only: the SAE layer files the script reads and the 9B base
# checkpoint, so the GPU runs start without waiting on downloads. The script re-verifies sha256 itself.
from huggingface_hub import hf_hub_download, snapshot_download
for repo, layer in (("Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50", 11), ("Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100", 11),
                    ("Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50", 15), ("Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100", 15)):
    hf_hub_download(repo, "config.json"); print(hf_hub_download(repo, f"layer{layer}.sae.pt"), flush=True)
print(snapshot_download("Qwen/Qwen3.5-9B-Base", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]), flush=True)
print("PREFETCH DONE", flush=True)
