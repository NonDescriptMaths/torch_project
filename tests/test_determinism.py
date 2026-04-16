import torch
import pytest
from src.model import train_one_step

def test_gradient_requirement(untrained_model):
    """Ensure model parameters actually require gradients."""
    for param in untrained_model.parameters():
        assert param.requires_grad is True

def test_gradients_actually_computed(untrained_model, sample_batch, sample_targets, optimizer_factory, criterion):
    """Verify that a training step actually populates the .grad attribute."""
    optimizer = optimizer_factory(untrained_model)
    
    # Run one step
    train_one_step(untrained_model, sample_batch, sample_targets, optimizer, criterion)
    
    # Check if gradients exist for the weight and bias
    for name, param in untrained_model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient!"
        assert torch.abs(param.grad).sum() > 0, f"Parameter {name} has a zero gradient!"

def test_overfit_small_batch(untrained_model, optimizer_factory, criterion):
    """
    Industry Standard: A model should be able to 'memorize' 
    a tiny dataset to near-zero loss.
    """
    optimizer = optimizer_factory(untrained_model, lr=0.1) # High LR for speed
    X = torch.randn(2, 10)
    y = torch.randn(2, 1)
    
    initial_loss = train_one_step(untrained_model, X, y, optimizer, criterion)
    
    # Train for 50 steps
    for _ in range(50):
        final_loss = train_one_step(untrained_model, X, y, optimizer, criterion)
        
    assert final_loss < initial_loss
    assert final_loss < 0.01 # It should basically solve it

def test_model_determinism(untrained_model, sample_batch):
    """Same input must give same output (crucial for inference)."""
    # Ensure model is in eval mode (standard practice)
    untrained_model.eval() 
    
    with torch.no_grad():
        out1 = untrained_model(sample_batch)
        out2 = untrained_model(sample_batch)
        
    assert torch.equal(out1, out2)