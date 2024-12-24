from models.visual_disambiguation_model import VisualDisambiguationModel
from data.dataset import ImagePairsDataset
from utils.image_utils import load_image
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

def main():
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = ImagePairsDataset(data_folder="train_set_noflip")
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Load DINOv2 backbone
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)

    # Initialize the model
    model = VisualDisambiguationModel(backbone=backbone).to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCELoss()

    # Training loop
    for epoch in range(5):  # Number of epochs
        model.train()
        for img1, img2, labels in data_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(img1, img2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/5], Loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()