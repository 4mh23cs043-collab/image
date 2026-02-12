import torch.nn as nn # type: ignore
from torchvision import models # type: ignore

def get_model(num_classes, weights=None):
    """
    Returns a MobileNetV2 model with a custom classifier head.
    If weights are provided (e.g., MobileNet_V2_Weights.DEFAULT), 
    the base model will be initialized with those weights.
    """
    if weights:
        model = models.mobilenet_v2(weights=weights)
        # Freeze base model
        for param in model.parameters():
            param.requires_grad = False
    else:
        # For loading custom weights later, we can use weights=None or pretrained=False (deprecated)
        # Using weights=None is the modern way.
        model = models.mobilenet_v2(weights=None)
    
    # Replace the default classifier with our custom one
    model.classifier[1] = nn.Sequential(
        nn.Linear(model.last_channel, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes)
    )
    
    return model
