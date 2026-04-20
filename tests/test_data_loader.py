import torch
from torch_project.data_cleaner import get_data_loaders, rescale_data

def test_get_data_loaders_logic():
    # 1. Create dummy data
    X = torch.randn(100, 5) # 100 rows
    y = torch.randn(100, 1)
    batch_size = 32
    
    # 2. Run your function
    loader = get_data_loaders(X, y, batch_size=batch_size)
    
    # 3. Assertions
    # Check if it's actually a DataLoader
    assert isinstance(loader, torch.utils.data.DataLoader)
    
    # Check if the first batch has the right size
    first_X, first_y = next(iter(loader))
    assert first_X.shape[0] == batch_size
    assert first_y.shape[0] == batch_size
    
    # Check total number of batches (100 / 32 = 4 batches)
    assert len(loader) == 4


def test_rescale_data_dimensions():
    # Mock some data
    X_raw = torch.randn(10, 5).numpy()
    y_raw = torch.randn(10, 1).numpy()
    
    X_tensor, y_tensor = rescale_data(X_raw, y_raw)
    
    # Test 1: Type check
    assert isinstance(X_tensor, torch.Tensor)
    # Test 2: Shape check
    assert X_tensor.shape == (10, 5)
    # Test 3: Mean check (Standardization should result in ~0)
    assert torch.abs(X_tensor.mean()) < 1e-5