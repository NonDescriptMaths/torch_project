import torch
import os
import json
from torch_project.data_cleaner import get_prepared_data, get_data_loaders
from torch_project.model import LinearModel, NonLinearModel
from torch_project.configs import TrainConfig
from torch_project.trainer import fit_model, evaluate_model
from torch_project.visualiser import plot_loss_curves
from torch_project.visualiser import plot_rate_vs_mse   

config = TrainConfig(
        name="diabetes_0.0001",
        model_type="nonlinear",
        lr=0.0001,
        epochs=100,
        batch_size=32
    )

def run_diabetes_trial(config=config):
    # 1. Setup
    X_train, X_test, y_train, y_test = get_prepared_data(test_size=0.2)
    
    loader = get_data_loaders(X_train, y_train, batch_size=config.batch_size)
    if config.model_type == "linear":
        model = LinearModel(input_dim=X_train.shape[1])
    else:
        model = NonLinearModel(input_dim=X_train.shape[1], hidden_dim=config.hidden_dim)

    # 2. Execute
    print(f"🚀 Training {config.name}...")
    trained_model, history = fit_model(model, loader, config)

    test_mse = evaluate_model(trained_model, X_test, y_test)
    print(f"📊 Training Final MSE: {history[-1]:.4f}")
    print(f"🏆 TEST MSE (Real Performance): {test_mse:.4f}")
    # 3. Persistence
    output_path = f"results/experiments/{config.name}"
    os.makedirs(output_path, exist_ok=True)
    
    torch.save(trained_model.state_dict(), f"{output_path}/weights.pth")
    
    with open(f"{output_path}/metrics.json", "w") as f:
        json.dump({
            "config": config.__dict__,
            "final_loss": history[-1],
            "history": history,
            "test_mse": test_mse
        }, f, indent=4)

if __name__ == "__main__":
    run_diabetes_trial(config)
    plot_loss_curves(config.name)
    plot_rate_vs_mse()