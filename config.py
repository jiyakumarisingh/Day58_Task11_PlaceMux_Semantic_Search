from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "artifacts"
LOG_DIR = BASE_DIR / "logs"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 10
