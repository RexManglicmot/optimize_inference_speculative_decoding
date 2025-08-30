# app/config.py

import os
import yaml
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env (for HF_API_KEY etc.)
load_dotenv()

# Path to config.yaml (one directory up from app/)
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

def load_config(path: Path = CONFIG_PATH):
    """Load YAML config into a dictionary."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

# Global config object
cfg = load_config()

# Convenience: grab HF key directly
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
print(HF_API_TOKEN)

if HF_API_TOKEN is None:
    raise ValueError(
        "Hugging Face API key not found. Please add HF_API_TOKEN=... to your .env file."
    )

# DONE
# Can print out config...do it later....