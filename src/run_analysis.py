from src.data_loader import get_prepared_data
from src.model import LinearModel, train_one_step
from src.visualiser import plot_predictions
import torch

# 1. Setup
X, y = get_prepared_data()
model = LinearModel(input_dim=X.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

# 2. Training Loop
print("Training...")
for epoch in range(200):
    loss = train_one_step(model, X, y, optimizer, criterion)
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss {loss:.4f}")

# 3. Look at results
plot_predictions(model, X, y)