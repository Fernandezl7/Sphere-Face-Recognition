import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from mydatasets.celebA_dataset import CelebADataset, load_identity_mapping
from models.sphereface_model import SphereFaceNet


def main():
    # ---------------------
    # Paths
    # ---------------------
    identity_txt_path = 'data/identity_CelebA.txt'
    img_dir = 'data/img_align_celeba'

    if not os.path.exists(identity_txt_path) or not os.path.isdir(img_dir):
        raise FileNotFoundError("Missing identity file or image folder.")

    # ---------------------
    # Load + remap labels for subset
    # ---------------------
    identity_dict = load_identity_mapping(identity_txt_path)

    # Limit to first 500 images
    limited_keys = list(identity_dict.keys())[:500]

    # Extract original IDs and remap to 0-based
    original_labels = sorted(set(identity_dict[k] for k in limited_keys))
    label_map = {orig_id: new_id for new_id, orig_id in enumerate(original_labels)}
    remapped_identity_dict = {k: label_map[identity_dict[k]] for k in limited_keys}

    # Dataset and DataLoader
    dataset = CelebADataset(img_dir, remapped_identity_dict)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # ---------------------
    # Model, Loss, Optimizer
    # ---------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(label_map)

    model = SphereFaceNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # ---------------------
    # Training Loop
    # ---------------------
    EPOCHS = 1  # Use 1 epoch for testing
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)

            logits, labels = model(imgs, labels)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss:.4f}")

    # ---------------------
    # Save model
    # ---------------------
    save_path = 'saved_models/sphereface_model.pth'
    os.makedirs('saved_models', exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")


if __name__ == '__main__':
    main()
