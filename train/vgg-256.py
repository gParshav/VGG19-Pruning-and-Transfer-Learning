import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from custom_vgg import CustomVGG19

# Configuration
data_dir = 'data/256_ObjectCategories'  # Replace with the path to your 101_ObjectCategories folder
batch_size = 128
num_epochs = 10  # Adjust as needed
learning_rate = 0.001
num_classes = 257  # Caltech 256 has 256 categories + 1 background category
train_ratio = 0.8  # 10% for training, 90% for testing

# Check if MPS (Mac GPU) is available; else use CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 (VGG-19 input size)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Loading the dataset and splitting into train and test sets
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the model and set it to the device (GPU or CPU)
model = CustomVGG19(num_classes=num_classes)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop with accuracy printing
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update loss
        running_loss += loss.item()
        
        # Compute accuracy
        _, predicted = torch.max(outputs, 1)
        running_corrects += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        # Print stats every 10 steps
        if (i + 1) % 10 == 0:
            batch_accuracy = 100 * running_corrects / total_samples
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], "
                  f"Loss: {loss.item():.4f}, Accuracy: {batch_accuracy:.2f}%")
    
    # Print epoch stats
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * running_corrects / total_samples
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")


# Testing loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate and print accuracy
accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# Save the trained model
torch.save(model.state_dict(), 'weights/vgg19_custom_caltech256.pth')
print("Model saved to weights/vgg19_custom_caltech256.pth")