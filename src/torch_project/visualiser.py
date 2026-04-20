import torch
import matplotlib.pyplot as plt
import matplotlib

# Use Agg backend to generate files without a GUI
matplotlib.use('Agg') 

def plot_predictions(model, X, y):
    model.eval()
    with torch.no_grad():
        predictions = model(X)
    
    # Take first feature for 2D plotting
    x_plot = X[:, 0].numpy()
    y_true = y.numpy()
    y_pred = predictions.numpy()

    plt.figure(figsize=(10, 6))
    plt.scatter(x_plot, y_true, label='Actual', alpha=0.5)
    plt.scatter(x_plot, y_pred, label='Predicted', color='red')
    plt.legend()
    
    # Save instead of show to avoid FigureCanvasAgg error
    plt.savefig('predictions.png')
    plt.close()