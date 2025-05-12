import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from models.sphereface_model import SphereFaceNet


# -------------------------------
# Load the trained model
# -------------------------------
def load_model(model_path, num_classes=500):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SphereFaceNet(num_classes=num_classes).to(device)

    # Load checkpoint while ignoring classifier mismatch
    checkpoint = torch.load(model_path, map_location=device)
    model_dict = model.state_dict()

    # Filter out classifier weights (we don't need them for embedding)
    filtered_dict = {k: v for k, v in checkpoint.items() if k in model_dict and "classifier" not in k}
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict, strict=False)
    model.eval()

    print("✅ Model loaded for inference (ignoring classifier mismatch)")
    return model, device


# -------------------------------
# Preprocess a single image
# -------------------------------
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((112, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # add batch dimension


# -------------------------------
# Extract face embedding
# -------------------------------
def get_embedding(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        embedding = model(image_tensor)  # returns 512-D normalized vector
    return embedding.squeeze(0)

def extract_features(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        # Manually run feature extractor and fc
        x = model.features(image_tensor)
        x = x.view(x.size(0), -1)
        embedding = model.fc(x)
        return F.normalize(embedding, p=2, dim=1).squeeze(0)



# -------------------------------
# Cosine similarity
# -------------------------------
def cosine_similarity(emb1, emb2):
    return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()


# -------------------------------
# Main
# -------------------------------
if __name__ == '__main__':
    model_path = 'saved_models/sphereface_model.pth'
    image1_path = 'data/img_align_celeba/000001.jpg'
    image2_path = 'data/img_align_celeba/000002.jpg'

    model, device = load_model(model_path, num_classes=500)

    img1 = preprocess_image(image1_path)
    img2 = preprocess_image(image2_path)

    emb1 = get_embedding(model, img1, device)
    emb2 = get_embedding(model, img2, device)

    sim = cosine_similarity(emb1, emb2)
    print(f"Cosine similarity between images: {sim:.4f}")
