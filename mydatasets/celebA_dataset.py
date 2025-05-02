import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def load_identity_mapping(identity_file_path):
    """
    Load image-to-identity mappings from identity_CelebA.txt.
    Returns a dictionary: {filename: label_id}
    """
    identity_dict = {}
    with open(identity_file_path, 'r') as f:
        for line in f:
            filename, identity = line.strip().split()
            identity_dict[filename] = int(identity)
    return identity_dict


def default_transform():
    """
    Default image preprocessing: resize, normalize
    """
    return transforms.Compose([
        transforms.Resize((112, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])


class CelebADataset(Dataset):
    """
    Custom PyTorch Dataset for CelebA face recognition.
    """
    def __init__(self, img_dir, identity_dict, transform=None):
        self.img_dir = img_dir
        self.identity_dict = identity_dict
        self.transform = transform if transform else default_transform()
        self.filenames = list(identity_dict.keys())

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert('RGB')
        label = self.identity_dict[filename]  #No `-1` here!
        image = self.transform(image)
        return image, label
