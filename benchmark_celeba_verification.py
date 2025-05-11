import os
import random
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from inference import load_model, get_embedding, cosine_similarity

# Set paths
IMG_DIR = "data/img_align_celeba"
ID_FILE = "data/identity_CelebA.txt"
MODEL_PATH = "saved_models/sphereface_model_final.pth"

# Hyperparameters
NUM_PAIRS = 500
IMAGE_SIZE = 112

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

def load_identities(id_file):
    with open(id_file, 'r') as f:
        lines = f.readlines()
    mapping = {}
    for line in lines:
        img, pid = line.strip().split()
        if pid not in mapping:
            mapping[pid] = []
        mapping[pid].append(img)
    return mapping

def sample_pairs(identity_map, num_pairs):
    same_pairs, diff_pairs = [], []
    # Sample same-identity pairs
    for pid, imgs in identity_map.items():
        if len(imgs) < 2:
            continue
        pairs = list(zip(imgs[:-1], imgs[1:]))
        same_pairs.extend(pairs[:num_pairs - len(same_pairs)])
        if len(same_pairs) >= num_pairs:
            break
    # Sample different-identity pairs
    pids = list(identity_map.keys())
    while len(diff_pairs) < num_pairs:
        pid1, pid2 = random.sample(pids, 2)
        img1 = random.choice(identity_map[pid1])
        img2 = random.choice(identity_map[pid2])
        diff_pairs.append((img1, img2))
    return same_pairs, diff_pairs

def preprocess(img_path):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    img = Image.open(img_path).convert('L')
    img = Image.open(img_path).convert('RGB')
    return transform(img).unsqueeze(0)  # shape: (1, 1, H, W)

def evaluate_model(model, same_pairs, diff_pairs):
    y_true = []
    y_score = []

    all_pairs = [(a, b, 1) for a, b in same_pairs] + [(a, b, 0) for a, b in diff_pairs]
    for img1_name, img2_name, label in tqdm(all_pairs, desc="Evaluating"):
        path1 = os.path.join(IMG_DIR, img1_name)
        path2 = os.path.join(IMG_DIR, img2_name)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        img1 = preprocess(path1).to(device)
        img2 = preprocess(path2).to(device)

        with torch.no_grad():
            emb1 = get_embedding(model, img1, device)
            emb2 = get_embedding(model, img2, device)

        sim = cosine_similarity(emb1, emb2)
        y_true.append(label)
        y_score.append(sim)

    # Compute metrics
    threshold = 0.5
    y_pred = [1 if s >= threshold else 0 for s in y_score]
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    return acc, auc

def main():
    print("Loading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_model(MODEL_PATH)
    model = model.to(device)
    model.eval()

    print("Loading identities...")
    identity_map = load_identities(ID_FILE)
    same_pairs, diff_pairs = sample_pairs(identity_map, NUM_PAIRS)

    print("Running evaluation...")
    acc, auc = evaluate_model(model, same_pairs, diff_pairs)
    print(f"Verification Accuracy: {acc * 100:.2f}%")
    print(f"ROC AUC Score: {auc:.4f}")

if __name__ == '__main__':
    main()
