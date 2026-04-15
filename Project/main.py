import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import confusion_matrix
from models import initialize_models

from dataset import get_data
from training import train_model
from utils import plot_fitting_curves, plot_confusion_matrix, save_data_checkpoint

# 1. Inject your Kaggle API token directly into the environment
os.environ['KAGGLE_API_TOKEN'] = "KGAT_f04e9f69e859b8117d064171adfe8a10"

import kaggle

def setup_dataset():
    dataset_name = "abhikalpsrivastava15/space-images-category"
    target_dir = "./data"

    if not os.path.exists(target_dir) or not os.listdir(target_dir):
        print("Downloading Astronomy dataset from Kaggle...")
        os.makedirs(target_dir, exist_ok=True)

        # This will now use the KGAT token you provided above!
        kaggle.api.dataset_download_files(dataset_name, path=target_dir, unzip=True)

        print("Dataset successfully downloaded and extracted into /data!")
    else:
        print("Dataset already exists locally. Skipping download.")

if __name__ == "__main__":

    setup_dataset()

    train_loader, val_loader, class_names = get_data()
    resnet, googlenet = initialize_models(len(class_names))
    criterion = nn.CrossEntropyLoss()

    optimizer_resnet = optim.SGD(resnet.parameters(), lr=0.001)
    optimizer_googlenet = optim.SGD(googlenet.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model, history = train_model(
        model=resnet,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_resnet,
        num_epochs=10,
        device=device
    )

    save_data_checkpoint(history, filename="resnet_history.json")

    plot_fitting_curves(history, model_name="ResNet18", filename="resnet_training_curves.png")

    y_true = []
    y_pred = []

    model.eval()

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu()

            y_pred.extend(preds.tolist())
            y_true.extend(labels.tolist())

    plot_confusion_matrix(
        y_true= y_true,
        y_pred= y_pred,
        class_names=class_names,
        model_name="ResNet18",
        filename="resnet_confusion_matrix.png"
    )

    model, history = train_model(
        model=googlenet,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_googlenet,
        num_epochs=10,
        device=device
    )

    save_data_checkpoint(history, filename="googlenet_history.json")

    plot_fitting_curves(history, model_name="GoogLeNet", filename="googlenet_training_curves.png")

    y_true = []
    y_pred = []

    model.eval()

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu()

            y_pred.extend(preds.tolist())
            y_true.extend(labels.tolist())

    plot_confusion_matrix(
        y_true=[label for _, label in val_loader.dataset],
        y_pred=[torch.argmax(model(inputs.to(device))).item() for inputs, _ in val_loader],
        class_names=class_names,
        model_name="GoogLeNet",
        filename="googlenet_confusion_matrix.png"
    )

