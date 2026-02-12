import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torchvision import datasets, models, transforms # type: ignore
from torch.utils.data import DataLoader # type: ignore
import matplotlib # type: ignore
matplotlib.use('Agg')
import matplotlib.pyplot as plt # type: ignore

from classifier_module import get_model # type: ignore
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
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset path '{DATASET_PATH}' not found.")
        return

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

    # Build Model
    print("Building model (PyTorch MobileNetV2)...")
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = get_model(num_classes, weights=weights)

    device = torch.device("cpu") # Explicitly use CPU for stability in this environment
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    # Training Loop
    print(f"Starting training on {device}...")
    history = {
        'train_acc': [],
        'val_acc': [],
        'train_loss': [],
        'val_loss': []
    }

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
        train_loss = running_loss / len(train_loader)
        history['train_acc'].append(train_acc) # type: ignore
        history['train_loss'].append(train_loss) # type: ignore

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        val_loss = val_running_loss / len(val_loader)
        history['val_acc'].append(val_acc) # type: ignore
        history['val_loss'].append(val_loss) # type: ignore
        
        print(f"Epoch {epoch+1} Summary: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Loss: {train_loss:.4f}")

    # Save Model
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    # Save classes mapping
    with open("classes.txt", "w") as f:
        f.write("\n".join(full_dataset.classes))

    # Plot and save history
    print("Generating training history plots...")
    epochs_range = range(1, EPOCHS + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(list(epochs_range), history['train_acc'], label='Train Acc', marker='o') # type: ignore
    plt.plot(list(epochs_range), history['val_acc'], label='Val Acc', marker='o') # type: ignore
    plt.title('Accuracy History')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(list(epochs_range), history['train_loss'], label='Train Loss', marker='o', color='red') # type: ignore
    plt.plot(list(epochs_range), history['val_loss'], label='Val Loss', marker='o', color='orange') # type: ignore
    plt.title('Loss History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved as training_history.png")

    print("Training task complete.")

if __name__ == "__main__":
    train_model()
