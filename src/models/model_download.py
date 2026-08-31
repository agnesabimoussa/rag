from pathlib import Path
from huggingface_hub import snapshot_download


class ModelDownload:
    @staticmethod
    def _ensure_local_weights(model_path: str) -> str:
        slug = model_path.split("/")[-1].lower()
        local_dir = Path("models") / slug
        if not (local_dir / "config.json").is_file():
            snapshot_download(repo_id=model_path, local_dir=str(local_dir))
        return str(local_dir)
