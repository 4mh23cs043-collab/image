import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# Configuration
DATASET_PATH = r"c:\image\Final Random Image Dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16 # Smaller batch size for CPU
EPOCHS = 3 # Fewer epochs for quick demonstration
MODEL_SAVE_PATH = "classifier_model.pth"
SUBSET_SIZE = 400 # Small subset for rapid training on CPU

def train_model():
    print("Preparing data...")
    # Transforms
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load dataset
    full_dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
    
    # Use a subset for faster demonstration
    indices = torch.randperm(len(full_dataset))[:SUBSET_SIZE]
    subset_dataset = torch.utils.data.Subset(full_dataset, indices)

    # Split into train and val
    train_size = int(0.8 * len(subset_dataset))
    val_size = len(subset_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(subset_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Dataset subset size: {len(subset_dataset)}")
    print(f"Classes: {full_dataset.classes}")
    num_classes = len(full_dataset.classes)

    # Build Model (Transfer Learning with MobileNetV2)
    print("Building model (PyTorch MobileNetV2)...")
    # Use weights instead of deprecated pretrained
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    
    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace classifier
    model.classifier[1] = nn.Sequential(
        nn.Linear(model.last_channel, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes)
    )

    device = torch.device("cpu") # Explicitly use CPU for stability in this environment
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    # Training Loop
    print(f"Starting training on {device}...")
    train_acc_history = []
    val_acc_history = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"Epoch {epoch+1}/{EPOCHS} started...")
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (i+1) % 5 == 0:
                print(f"  Step [{i+1}/{len(train_loader)}], Loss: {running_loss/(i+1):.4f}")

        train_acc = 100. * correct / total
        train_acc_history.append(train_acc)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        val_acc_history.append(val_acc)
        print(f"Epoch {epoch+1} Summary: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    # Save Model
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    # Save classes mapping
    with open("classes.txt", "w") as f:
        f.write("\n".join(full_dataset.classes))

    print("Training task complete.")

if __name__ == "__main__":
    train_model()
