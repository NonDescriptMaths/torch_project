import matplotlib.pyplot as plt
import json
import os

def plot_loss_curves(exp_name):
    """Plots the training history from the metrics.json file."""
    path = f"results/experiments/{exp_name}/metrics.json"
    
    with open(path, "r") as f:
        data = json.load(f)
    
    history = data["history"]
    
    plt.figure(figsize=(10, 5))
    plt.plot(history, label="Training Loss (MSE)")
    plt.title(f"Convergence: {exp_name}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    output_path = f"results/plots/{exp_name}"
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(f"results/plots/{exp_name}/convergence_plot.png")
    print(f"📈 Plot saved to results/plots/{exp_name}/convergence_plot.png")


def plot_rate_vs_mse():
    """Plots the final MSE for different learning rates."""
    rates = []
    mses = []
    
    base_path = "results/experiments"
    for exp in os.listdir(base_path):
        with open(f"{base_path}/{exp}/metrics.json", "r") as f:
            data = json.load(f)
            rates.append(data["config"]["lr"])
            mses.append(data["test_mse"])
    
    plt.figure(figsize=(10, 5))
    plt.plot(rates, mses, marker='o',linestyle = '')
    plt.xscale('log')
    plt.title("Learning Rate vs Test MSE")
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Test MSE")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    output_path = "results/plots/rate_vs_mse"
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(f"{output_path}/rate_vs_mse.png")
    print(f"📈 Plot saved to {output_path}/rate_vs_mse.png")