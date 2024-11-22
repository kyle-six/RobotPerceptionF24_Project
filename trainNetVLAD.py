import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from netVlad import NetVLAD
from torchvision.models import resnet18
import sys


def print_loading_bar(
    iteration, total, prefix="", suffix="", length=50, fill="█", print_end="\r"
):

    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + "-" * (length - filled_length)
    sys.stdout.write(f"\r{prefix} |{bar}| {percent}% {suffix}")
    sys.stdout.flush()

    if iteration == total:
        sys.stdout.write("\n")


# Define dataset class
class MazeDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_list = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_list[int(idx)])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, idx


# Define the triplet loss function
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, positive, negative):
        return self.loss(anchor, positive, negative)


# Data preparation
image_dir = "data/midterm_data/images/"

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize to match input size of CNN
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.6537, 0.6355, 0.6409], std=[0.3719, 0.3697, 0.3589]
        ),
    ]
)


dataset = MazeDataset(image_dir, transform)
batch_size = 30
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# mean = 0.0
# std = 0.0
# for images, _ in dataloader:
#     batch_samples = images.size(0)  # Get the number of samples in this batch
#     images = images.view(batch_samples, images.size(1), -1)
#     mean += images.mean(2).sum(0)
#     std += images.std(2).sum(0)

# mean /= len(dataloader.dataset)
# std /= len(dataloader.dataset)
# print(mean)
# print(std)


encoder = resnet18(pretrained=True)

base_model = nn.Sequential(
    encoder.conv1,
    encoder.bn1,
    encoder.relu,
    encoder.maxpool,
    encoder.layer1,
    encoder.layer2,
    encoder.layer3,
    encoder.layer4,
)
dim = list(base_model.parameters())[-1].shape[0]
net_vlad = NetVLAD(num_clusters=32, dim=dim, alpha=1.0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = base_model.to(device)
net_vlad_model = net_vlad.to(device)

# Training setup
triplet_loss = TripletLoss(margin=1.0).to(device)
optimizer = optim.Adam(net_vlad_model.parameters(), lr=1e-4)
epochs = 10


def generate_triplets(anchor_idxs, data_len, max_frame_gap=10, negative_gap=500):
    """Generates triplets of anchor, positive, and negative indices."""
    positive_idxs = torch.zeros(batch_size)
    negative_idxs = torch.zeros(batch_size)
    for i, anchor_idx in enumerate(anchor_idxs):
        positive_idxs[i] = random.randint(
            max(anchor_idx - max_frame_gap, 0),
            min(anchor_idx + max_frame_gap, data_len - 1),
        )

        if anchor_idx > 5000:
            negative_idxs[i] = random.randint(0, anchor_idx - negative_gap)
        else:
            negative_idxs[i] = random.randint(anchor_idx + negative_gap, data_len - 1)
    return anchor_idxs, positive_idxs, negative_idxs


# Training loop
for epoch in range(epochs):
    net_vlad_model.train()
    running_loss = 0.0
    nr_samples = 0
    for i, (anchor_imgs, anchor_idxs) in enumerate(dataloader):
        print_loading_bar(
            i, len(dataloader), prefix="Progress", suffix="Complete", length=40
        )
        nr_samples += batch_size
        # Generate triplets
        anchor_idxs, positive_idxs, negative_idxs = generate_triplets(
            anchor_idxs, len(dataset)
        )
        # Load images for triplet
        anchor_imgs = anchor_imgs.to(device)
        positive_imgs = torch.stack([dataset[p][0] for p in positive_idxs]).to(device)
        negative_imgs = torch.stack([dataset[n][0] for n in negative_idxs]).to(device)

        anchors_after_resnet = base_model(anchor_imgs)
        positives_after_resnet = base_model(positive_imgs)
        negatives_after_resnet = base_model(negative_imgs)

        # Forward pass
        anchors_feat = net_vlad_model(anchors_after_resnet)
        positives_feat = net_vlad_model(positives_after_resnet)
        negatives_feat = net_vlad_model(negatives_after_resnet)

        # Compute loss
        loss = triplet_loss(anchors_feat, positives_feat, negatives_feat)
        running_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss /nr_samples:.4f}")

    # Save the trained model
    torch.save(net_vlad_model.state_dict(), "netvlad_maze.pth")
