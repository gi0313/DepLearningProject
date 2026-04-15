import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
from training import train_model

def get_data(data_dir='data', batch_size=32):
    """
        Loads the Kaggle images, applies the required mathematical transformations
        for ResNet/GoogleNet, and packages them into mini-batch DataLoaders.
        """

    #Define the Transformations
    data_transforms = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")), # Ensures 3 channel input
        transforms.Resize((224, 224)),  #Force every image to the same size
        transforms.ToTensor(),  #Convert pixel values from 0-255 to tensors between 0.0-1.0
        transforms.Normalize(  #Standard ImageNet normalization values
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    print(f"Loading images from ./{data_dir} ")
    full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)

    #80% Train, 20% Validation
    total_count = len(full_dataset)
    train_count = int(0.8 * total_count)
    val_count = total_count - train_count

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_count, val_count],
        generator=torch.Generator().manual_seed(42)  # Keeps the split consistent across runs
    )

    #Mini-Batch DataLoaders
    # This prevents your GPU memory from crashing by feeding images in chunks of 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Save the class names (e.g., ['Galaxies', 'Nebulae', 'Stars']) for our confusion matrix later
    class_names = full_dataset.classes

    print(f"Successfully loaded {total_count} total images across {len(class_names)} categories.")
    print(f"Training set: {train_count} images | Validation set: {val_count} images.")

    return train_loader, val_loader, class_names

# Quick test block to ensure it works when you run this file directly
if __name__ == "__main__":
    # Assuming your Kaggle data is unzipped into a folder called 'data'
    os.makedirs('data', exist_ok=True)
    print("Dataset module ready. Make sure your images are inside the 'data' folder!")
    train_loader, val_loader, class_names = get_data("data\space images", batch_size=32)