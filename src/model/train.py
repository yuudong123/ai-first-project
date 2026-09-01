from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.hydrotwin_pipeline import main_train


if __name__ == "__main__":
    main_train()
