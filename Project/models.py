import torch.nn as nn
from torchvision import datasets, transforms, models

def initialize_models(num_classes):
    # Load pretrained models
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    googlenet = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)

    # Freeze all layers first
    for param in resnet.parameters():
        param.requires_grad = False

    for param in googlenet.parameters():
        param.requires_grad = False

    # Replace final layer

    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

    googlenet.fc = nn.Linear(googlenet.fc.in_features, num_classes)

    # unfreeze the new layers so they can train
    for param in resnet.fc.parameters():
        param.requires_grad = True

    for param in googlenet.fc.parameters():
        param.requires_grad = True

    return resnet, googlenet