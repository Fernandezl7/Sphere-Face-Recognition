import torch
from models.sphereface_model import SphereFaceNet

def load_trained_model(num_classes, model_path='saved_models/sphereface_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SphereFaceNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Loaded model from: {model_path}")
    return model
