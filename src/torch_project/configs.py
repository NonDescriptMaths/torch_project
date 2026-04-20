from dataclasses import dataclass

@dataclass
class TrainConfig:
    name: str
    model_type: str  # "linear" or "mlp"
    lr: float
    epochs: int
    batch_size: int
    hidden_dim: int = 64