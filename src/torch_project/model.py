import torch.nn as nn
import torch.nn.functional as F

class LinearModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)
    
class NonLinearModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        # Layer 1: Input to Hidden
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Layer 2: Hidden to Hidden
        self.fc2 = nn.Linear(hidden_dim, 32)
        # Layer 3: Hidden to Output
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x)) # Activation function
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_one_step(model, X, y, optimizer, criterion):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    return loss.item()