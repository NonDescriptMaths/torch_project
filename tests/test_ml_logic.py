import torch
import torch.nn as nn
from torch_project.model import train_one_step

def test_model_output_shape(untrained_model, sample_batch):
    # Act
    expected_batch_size = sample_batch.shape[0]

    output = untrained_model(sample_batch)
    
    # Assert: Output should be (batch_size, 1)
    assert output.shape == (expected_batch_size, 1)

def test_training_step_reduces_loss(untrained_model):
    # Setup
    X = torch.randn(10, 10)
    y = torch.randn(10, 1)
    optimizer = torch.optim.SGD(untrained_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Act
    loss_before = train_one_step(untrained_model, X, y, optimizer, criterion)
    loss_after = train_one_step(untrained_model, X, y, optimizer, criterion)
    
    # Assert: Optimization should generally reduce loss (or at least change it)
    assert loss_after != loss_before