import torch
from sklearn.datasets import load_diabetes

def get_prepared_data():
    data = load_diabetes()
    # Features (X) and Target (y)
    X = torch.tensor(data.data, dtype=torch.float32)
    y = torch.tensor(data.target, dtype=torch.float32).view(-1, 1)
    return X, y

