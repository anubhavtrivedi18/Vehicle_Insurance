import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipline.training_pipeline import TrainPipeline


if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()