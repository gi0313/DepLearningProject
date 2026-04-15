import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import seaborn as sns
import json

#Saves the training history to prevent data loss if program crashes
def save_data_checkpoint(data_dict, filename="model_history.json"):

    filepath = f"plots/{filename}"
    with open(filepath, 'w') as f:
        json.dump(data_dict, f, indent=4)
    print(f"--> Data safely backed up to {filepath}")

os.makedirs('plots', exist_ok=True)

def plot_fitting_curves(history, model_name="Model", filename="training_curves.png"):
    #Plots the training and test proxy error in one graph training and misclassification together on a separate graph.
    epochs = range(1, len(history['train_proxy']) + 1)

    #Create a figure with two side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    #Cross-Entropy Loss
    ax1.plot(epochs, history['train_acc'], label='Training Accuracy', color='blue')
    ax1.plot(epochs, history['val_acc'], label='Validation Accuracy', color='orange', linestyle='--')
    ax1.set_title(f'{model_name} - Cross-Entropy Loss vs. Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    #Graph Accuracy
    ax1.plot(epochs, history['train_acc'], label='Training Accuracy', color='blue')
    ax2.plot(epochs, history['val_acc'], label='Validation Accuracy', color='orange', linestyle='--')
    ax2.set_title(f'{model_name} - Accuracy vs. Epochs')
    ax1.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (0.0 to 1.0)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f'plots/{filename}', bbox_inches='tight', dpi=300)
    plt.close()
    #plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, model_name="Model", filename="confusion_matrix.png"):
    #Generates a color-coded Seaborn heatmap showing exactly which space categories the model is confusing with each other

    #Calculate the raw numbers using scikit-learn
    cm = confusion_matrix(y_true, y_pred)

    #Seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)

    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('Actual Space Category')
    plt.xlabel('Predicted Space Category')

    plt.tight_layout()
    plt.savefig(f'plots/{filename}', bbox_inches='tight', dpi=300)
    plt.close()
    #plt.show()