import os
import random
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from inference import load_model, get_embedding, cosine_similarity
import cv2
import numpy as np

# Set paths
DATA_DIR = "data"
IMG_DIR = "data/img_align_celeba"
ID_FILE = "data/identity_CelebA.txt"
MODEL_PATH = 'saved_models/sphereface_model_sphereface_loss.pth'

# Hyperparameters
NUM_PAIRS = 500
IMAGE_SIZE = 112

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

def load_landmarks(landmark_path):
    landmarks = {}
    with open(landmark_path, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            parts = line.strip().split()
            filename = parts[0]
            coords = list(map(int, parts[1:]))
            points = np.array(coords, dtype=np.float32).reshape(-1, 2)
            landmarks[filename] = points
    return landmarks

def align_face(image, landmarks, output_size=(112, 112)):
    # Use left eye, right eye, nose tip
    src = np.array([
        landmarks[0],  # left eye
        landmarks[1],  # right eye
        landmarks[2]   # nose
    ], dtype=np.float32)

    dst = np.array([
        [38.2946, 51.6963],   # standard left eye
        [73.5318, 51.5014],   # standard right eye
        [56.0252, 71.7366]    # standard nose tip
    ], dtype=np.float32)

    tform = cv2.getAffineTransform(src, dst)
    aligned = cv2.warpAffine(np.array(image), tform, output_size, borderValue=0.0)
    return Image.fromarray(aligned)

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

def preprocess_image(image_path, landmarks_dict=None):
    image = Image.open(image_path).convert('RGB')

    if landmarks_dict:
        filename = os.path.basename(image_path)
        if filename in landmarks_dict:
            landmarks = landmarks_dict[filename]
            image = align_face(image, landmarks)

    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension


def evaluate_model(model, same_pairs, diff_pairs, landmarks_dict):
    y_true = []
    y_score = []

    all_pairs = [(a, b, 1) for a, b in same_pairs] + [(a, b, 0) for a, b in diff_pairs]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for img1_name, img2_name, label in tqdm(all_pairs, desc="Evaluating"):
        path1 = os.path.join(IMG_DIR, img1_name)
        path2 = os.path.join(IMG_DIR, img2_name)

        img1 = preprocess_image(path1, landmarks_dict).to(device)
        img2 = preprocess_image(path2, landmarks_dict).to(device)

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

    print("Loading landmarks...")
    landmarks_path = os.path.join(DATA_DIR, 'list_landmarks_align_celeba.txt')
    landmarks_dict = load_landmarks(landmarks_path)

    print("Running evaluation...")
    acc, auc = evaluate_model(model, same_pairs, diff_pairs, landmarks_dict)
    print(f"Verification Accuracy: {acc * 100:.2f}%")
    print(f"ROC AUC Score: {auc:.4f}")

if __name__ == '__main__':
    main()
