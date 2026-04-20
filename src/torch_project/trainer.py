import torch
from torch_project.model import train_one_step

def fit_model(model, loader, config):
    """
    Fits the model to the data provided by the loader.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = torch.nn.MSELoss()
    history = []

    model.train()
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            loss = train_one_step(model, batch_X, batch_y, optimizer, criterion)
            epoch_loss += loss
            
        avg_loss = epoch_loss / len(loader)
        history.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {avg_loss:.6f}")
            
    return model, history

def evaluate_model(model, X_test, y_test):
    """Calculates MSE on the test data specifically."""
    model.eval()
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        predictions = model(X_test)
        test_loss = criterion(predictions, y_test)
    return float(test_loss)