import torch
from torch_project.data_cleaner import get_prepared_data

def test_data_loader_properties():
    X, y = get_prepared_data()
    
    # Check types
    assert isinstance(X, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    
    # Check for NaNs (The silent killer of ML)
    assert not torch.isnan(X).any(), "Found NaNs in features!"
    assert not torch.isnan(y).any(), "Found NaNs in targets!"
    
    # Check alignment
    assert X.shape[0] == y.shape[0], "Features and targets have different row counts!"

