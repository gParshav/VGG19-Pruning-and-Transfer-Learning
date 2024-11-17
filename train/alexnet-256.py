import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from custom_alex import CustomAlexNet
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Configuration

data_dir = 'data/256_ObjectCategories'  # Replace with the path to your dataset
batch_size = 128
num_epochs = 50
learning_rate = 0.001
num_classes = 257
train_split_ratio = 0.8
val_split_ratio = 0.1
test_split_ratio = 0.1

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Split dataset into training, validation, and testing
train_size = int(train_split_ratio * len(dataset))
val_size = int(val_split_ratio * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model, loss, and optimizer
model = CustomAlexNet(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss, running_corrects, total_samples = 0.0, 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        running_corrects += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * running_corrects / total_samples
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Validation
    model.eval()
    val_loss, val_corrects, val_samples = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_corrects += (predicted == labels).sum().item()
            val_samples += labels.size(0)

    val_loss /= len(val_loader)
    val_accuracy = 100 * val_corrects / val_samples
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}% "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

# Save the model weights
torch.save(model.state_dict(), 'weights/custom_alexnet_weights2.pth')
print("Model weights saved to weights/custom_alexnet_weights2.pth")

print(train_losses)

print(val_losses)

print(train_accuracies)

print(val_accuracies)

# Plotting Loss and Accuracy
plt.figure(figsize=(12, 5))

# Loss curve
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.xticks(np.arange(1, num_epochs + 1, step=1))  # Set x-ticks to integer values

plt.subplot(1, 2, 1)
plt.savefig('plots/train_val_loss.png')

# Accuracy curve
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Curve')
plt.legend()
plt.xticks(np.arange(1, num_epochs + 1, step=1))  # Set x-ticks to integer values

# Save Accuracy curve
plt.subplot(1, 2, 2)
plt.savefig('plots/train_val_accuracy.png')

plt.tight_layout()
plt.show()

# Evaluate on test dataset and Confusion Matrix
model.eval()
test_corrects, test_samples = 0, 0
all_labels, all_predictions = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        test_corrects += (predicted == labels).sum().item()
        test_samples += labels.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

test_accuracy = 100 * test_corrects / test_samples
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_predictions)

# Create and save the confusion matrix plot
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=dataset.classes)
disp.plot(cmap='viridis', xticks_rotation='vertical')

# Save the confusion matrix plot
plt.title('Confusion Matrix')
plt.savefig('plots/confusion_matrix.png')

# Display the confusion matrix
plt.show()