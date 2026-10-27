"""Download only the official, pinned original Gemma checkpoint."""
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def main():
    config = json.loads(Path("config.json").read_text())
    target = Path(config["model_path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    info = HfApi(token=False).model_info(config["model_id"], revision=config["revision"],
                                       files_metadata=True)
    if info.sha != config["revision"]:
        raise RuntimeError("Checkpoint revision mismatch")
    patterns = ("config.json", "generation_config.json", "tokenizer.json",
                "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model",
                "chat_template.jinja", "model.safetensors.index.json")
    selected = [f for f in info.siblings
                if f.rfilename in patterns or f.rfilename.endswith(".safetensors")]
    total = sum(f.size or 0 for f in selected)
    free = shutil.disk_usage(target.parent).free
    if free < total + 50 * 1024**3:
        raise RuntimeError("Insufficient disk: checkpoint size plus 50 GiB reserve required")
    print(json.dumps({"revision": info.sha, "download_bytes": total,
                      "disk_free_bytes": free}), flush=True)
    snapshot_download(config["model_id"], revision=config["revision"], token=False,
                      local_dir=target, allow_patterns=[f.rfilename for f in selected],
                      max_workers=2)
    (target / "origin.json").write_text(json.dumps({
        "model_id": info.id, "revision": info.sha, "weight_bytes": total,
        "files": {f.rfilename: f.size for f in selected},
    }, indent=2) + "\n")
    print("Pinned original checkpoint downloaded.", flush=True)


if __name__ == "__main__":
    main()
