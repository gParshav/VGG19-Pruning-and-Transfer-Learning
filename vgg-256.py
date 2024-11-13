import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Configuration
data_dir = '256_ObjectCategories'  # Replace with the path to your 101_ObjectCategories folder
batch_size = 32
num_epochs = 10  # Adjust as needed
learning_rate = 0.001
num_classes = 102  # Caltech 101 has 101 categories + 1 background category

# Check if MPS (Mac GPU) is available; else use CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 (VGG-19 input size)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Loading the dataset
train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define VGG-19 model from scratch
class CustomVGG19(nn.Module):
    def __init__(self, num_classes=102):
        super(CustomVGG19, self).__init__()
        
        # Define the VGG-19 convolutional layers
        self.features = nn.Sequential(
            # Conv Layer 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 3x224x224 -> 64x224x224
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 64x224x224 -> 64x224x224
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x224x224 -> 64x112x112
            
            # Conv Layer 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 64x112x112 -> 128x112x112
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 128x112x112 -> 128x112x112
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128x112x112 -> 128x56x56
            
            # Conv Layer 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 128x56x56 -> 256x56x56
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 256x56x56 -> 256x56x56
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 256x56x56 -> 256x56x56
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 256x56x56 -> 256x28x28
            
            # Conv Layer 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # 256x28x28 -> 512x28x28
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 512x28x28 -> 512x28x28
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 512x28x28 -> 512x28x28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 512x28x28 -> 512x14x14
            
            # Conv Layer 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 512x14x14 -> 512x14x14
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 512x14x14 -> 512x14x14
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 512x14x14 -> 512x14x14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 512x14x14 -> 512x7x7
        )
        
        # Define the fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 7 * 7, 4096),  # Flattened 512x7x7 -> 4096
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),  # 4096 -> 4096
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),  # 4096 -> num_classes
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten the feature maps for the fully connected layers
        x = self.classifier(x)
        return x

# Instantiate the model and set it to the device (GPU or CPU)
model = CustomVGG19(num_classes=num_classes)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# DataLoader for training dataset
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 10 == 0:  # Print every 10 batches
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    # Print epoch loss
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'vgg19_custom_caltech256.pth')
print("Model saved to vgg19_custom_caltech256.pth")