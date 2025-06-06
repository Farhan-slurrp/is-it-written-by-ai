from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # Data paths
    raw_data_dir: Path = Path("../data/raw")
    processed_data_dir: Path = Path("../data/processed")
    model_dir: Path = Path("../models")
    log_dir: Path = Path("experiments")

    # Model settings
    model_name: str = "roberta-base"
    max_length: int = 512
    num_labels: int = 2

    # Training hyperparameters
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 4

    # Random seed for reproducibility
    seed: int = 42

    # Checkpointing
    save_model: bool = True
    checkpoint_path: Path = model_dir / "best_model.pt"
