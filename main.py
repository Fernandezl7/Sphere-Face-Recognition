import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from mydatasets.celebA_dataset import CelebADataset, load_identity_mapping
from models.sphereface_model import SphereFaceNet
from losses.sphereface_loss import SphereFaceLoss

# ---------------------
# Weight Initialization
# ---------------------
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

# ---------------------
# Main Training Function
# ---------------------
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

    # Limit to first 500 images for quick training
    #limited_keys = list(identity_dict.keys())[:500]

    #limited_keys = list(identity_dict.keys())  # Use all available images

    # Limit training to 1000 identities, up to 20 images each
    MAX_IDENTITIES = 1000
    MAX_IMAGES_PER_ID = 20

    limited_keys = []
    identity_counts = {}

    for img_name, identity in identity_dict.items():
        if identity not in identity_counts:
            if len(identity_counts) >= MAX_IDENTITIES:
                continue
            identity_counts[identity] = 0

        if identity_counts[identity] < MAX_IMAGES_PER_ID:
            limited_keys.append(img_name)
            identity_counts[identity] += 1

    # Remap labels to 0-based index
    all_labels = sorted(set(identity_dict.values()))
    label_map = {orig_id: new_id for new_id, orig_id in enumerate(all_labels)}
    remapped_identity_dict = {k: label_map[identity_dict[k]] for k in limited_keys}

    # Dataset and DataLoader
    dataset = CelebADataset(img_dir, remapped_identity_dict)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=2)

    # ---------------------
    # Model, Loss, Optimizer
    # ---------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(label_map)

    model = SphereFaceNet(num_classes=num_classes).to(device)
    initialize_weights(model)

    #criterion = SphereFaceLoss(m=4, s=30.0)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5, verbose=True)

    best_loss = float('inf')
    #save_path = 'saved_models/sphereface_model_sphereface_loss.pth'
    save_path = 'saved_models/sphereface_ce_1000ids.pth'
    os.makedirs('saved_models', exist_ok=True)

    # ---------------------
    # Training Loop
    # ---------------------
    EPOCHS = 50
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        for imgs, labels in progress_bar:
            imgs, labels = imgs.to(device), labels.to(device)

            # Model prediction
            output = model(imgs)
            
            # Ensure logits is a tensor, not a tuple
            if isinstance(output, tuple):
                logits, _ = output
            else:
                logits = output

            loss = criterion(logits, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # Accumulate loss
            total_loss += loss.item() * imgs.size(0)
            progress_bar.set_postfix(loss=loss.item())

            # Debug: Check predictions occasionally
            if epoch % 10 == 0:
                with torch.no_grad():
                    pred = torch.argmax(logits, dim=1)
                    print(f"Predictions: {pred[:5].cpu().numpy()}, Actual: {labels[:5].cpu().numpy()}")

        # Scheduler step at the end of the epoch
        scheduler.step(total_loss / len(dataset))

        # Print epoch summary
        avg_loss = total_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Average Loss: {avg_loss:.4f}")
        for param_group in optimizer.param_groups:
            print(f"Learning Rate: {param_group['lr']}")

        if epoch == 0 or epoch % 5 == 0:
            print(f"Batch loss: {loss.item():.4f}")


        # Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model improved, saved to: {save_path}")

    # ---------------------
    # Save Final Model
    # ---------------------
    torch.save(model.state_dict(), save_path)
    print(f"Final model saved to: {save_path}")

if __name__ == '__main__':
    main()
