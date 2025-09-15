# app/config.py

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Path to config.yaml (one level up from app/)
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

def load_config(path: Path = CONFIG_PATH):
    """Load YAML config into a dictionary."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # Normalize project dirs (expand ./data, ./outputs → absolute paths)
    root = path.parent
    if "project" in cfg:
        if "input_dir" in cfg["project"]:
            cfg["project"]["input_dir"] = str((root / cfg["project"]["input_dir"]).resolve())
        if "output_dir" in cfg["project"]:
            cfg["project"]["output_dir"] = str((root / cfg["project"]["output_dir"]).resolve())
    return cfg

# Global config object
cfg = load_config()

# Hugging Face API key (optional if using private models)
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Run this file directly to debug config
if __name__ == "__main__":
    print("✅ Config loaded successfully")
    print("Project name:", cfg["project"]["name"])
    print("Input dir:", cfg["project"]["input_dir"])
    print("Output dir:", cfg["project"]["output_dir"])
    print("Draft models:", [d["id"] for d in cfg["models"]["drafts"]])
    print("Verifier model:", cfg["models"]["verifier"]["id"])
    print("Batch size:", cfg["batching"]["batch_size"])
    print("HF_API_TOKEN found:", HF_API_TOKEN is not None)

# Run python3 -m app.config
# It worked!!