import torch
from sklearn.datasets import load_diabetes
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def get_raw_data():
    """
    Fetches the diabetes dataset from sklearn.
    Returns: X (numpy), y (numpy)
    """
    data = load_diabetes()
    X = data.data
    y = data.target.reshape(-1, 1) # Reshape for the scaler
    return X, y

def rescale_data(X_raw, y_raw):
    """
    Standardizes features and targets to Mean=0, Std=1.
    Returns: X_tensor, y_tensor
    """
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_x.fit_transform(X_raw)
    y_scaled = scaler_y.fit_transform(y_raw)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
    
    return X_tensor, y_tensor


def get_prepared_data(test_size=0.2):
    X_raw, y_raw = get_raw_data()
    X_tensor, y_tensor = rescale_data(X_raw, y_raw)
    
    # Split: 80% for training, 20% for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_tensor, y_tensor, test_size=test_size, random_state=42
    )
    
    return X_train, X_val, y_train, y_val
def get_data_loaders(X, y, batch_size=32):
    # 1. Wrap tensors into a Dataset object
    dataset = TensorDataset(X, y)
    
    # 2. Create the Loader
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True  # Always shuffle training data!
    )
    
    return loader