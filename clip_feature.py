import h5py 
import clip
import torch
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.preprocess(image)
        return image

def extract(image_paths, batch_size=128, num_workers=32):
    device = 'cuda'
    model, preprocess = clip.load('ViT-L/14', device=device)

    dataset = ImageDataset(image_paths, preprocess)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    features = []

    with torch.no_grad():
        for batch_images in tqdm(data_loader, desc='extract'):
            batch_images_tensor = batch_images.to(device)
            batch_features = model.encode_image(batch_images_tensor)
            features.append(batch_features.cpu())

    features = torch.cat(features, dim=0)

    return features

def get_names(filename):

    image_paths = []
    with open(filename, 'r') as file:
        for line in file:
            # Split each line by space and get the first element (image path)
            image_path = line.split()[0]
            image_paths.append(image_path)
    return image_paths  

if __name__ == '__main__':

    test_files = get_names('expr_test.txt')


    test_features = extract(test_files)
    print('test_features.shape =', test_features.shape)

    # Saving features into an HDF5 file
    with h5py.File('expr_test.h5', 'w') as h5f:  # Open or create an HDF5 file
        # Create datasets within the HDF5 file
        h5f.create_dataset('test_features', data=test_features)
