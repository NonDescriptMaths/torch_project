import pytest
import torch
import torch.nn as nn
from torch_project.model import LinearModel

@pytest.fixture
def input_dim():
    return 10

@pytest.fixture
def sample_batch(input_dim):
    # Returns (batch_size=8, features=10)
    return torch.randn(8, input_dim)

@pytest.fixture
def sample_targets():
    # Returns (batch_size=8, output=1)
    return torch.randn(8, 1)

@pytest.fixture
def untrained_model(input_dim):
    return LinearModel(input_dim=input_dim)

@pytest.fixture
def criterion():
    return nn.MSELoss()

@pytest.fixture
def optimizer_factory():
    """Returns a function to create an optimizer for a given model."""
    def _make_opt(model, lr=0.01):
        return torch.optim.SGD(model.parameters(), lr=lr)
    return _make_opt