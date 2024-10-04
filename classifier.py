import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn


# Custom Dataset class to load images and apply transformations
class TextureDataset(Dataset):
    def __init__(self, img_folder, transform=None, num_augmentations=75):
        self.img_folder = img_folder
        self.img_names = [img for img in os.listdir(img_folder) if img.endswith(".png")]
        self.transform = transform
        self.num_augmentations = num_augmentations

    def __len__(self):
        return (
            len(self.img_names) * self.num_augmentations
        )  # each image augmented multiple times

    def __getitem__(self, idx):
        img_idx = (
            idx // self.num_augmentations
        )  # calculate the index of the original image
        img_name = self.img_names[img_idx]
        img_path = os.path.join(self.img_folder, img_name)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)
        label = int(img_name.split("_")[-1].split(".")[0]) - 1
        # label = torch.tensor(label)

        return img, label


# Define transformations including random crops, flips, and stretches
transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            224, scale=(0.2, 0.5), ratio=(0.75, 1.33)
        ),  # Random crop
        transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of horizontal flip
        transforms.RandomAffine(
            degrees=10, scale=(0.9, 1.1), shear=5
        ),  # Random rotation, scale, and shear
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize for pretrained models
    ]
)

# Parameters
batch_size = 32
img_folder = "data/textures"

# Create dataset and dataloader
dataset = TextureDataset(img_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Load a pretrained ResNet model and modify it for your task
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 200),
    nn.Softmax(dim=1),  # Assuming binary classification, adjust accordingly
)
# Define optimizer, criterion (loss function), and training parameters
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
# Move model to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# Training loop
num_epochs = 10  # Adjust this based on your needs

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total_batches = len(dataloader)
    for i, (inputs, labels) in enumerate(dataloader):
        # inputs = inputs.to(device)
        # labels = labels.float().to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 10 == 0:  # Print every 10 batches
            avg_loss = running_loss / (i + 1)  # Average loss so far
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{total_batches}], Loss: {avg_loss:.4f}"
            )

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}")

# Save the trained model
torch.save(model.state_dict(), "texture_model.pth")

# Notify via email (hypothetical, needs an email system)
print("Training complete. Model saved and ready to be emailed.")
