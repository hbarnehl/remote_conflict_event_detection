from torch import from_numpy, tensor
from torch.utils.data import Dataset, DataLoader

# Other
import os
import numpy as np
import torch
from tqdm import tqdm as tqdm
from pandas import read_csv
# from osgeo import gdal
import rasterio

# Functions
def load_4band_image(path):
    """Load a 4-band image and ensure it has dimensions [1, channels, height, width]"""
    with rasterio.open(path) as src:
        # Read only needed bands directly into a single array
        img = np.stack([src.read(i) for i in [3, 2, 1]], axis=0)  # RGB order
    
    # Normalize the image data to 0-1 range for display
    img_min, img_max = img.min(), img.max()
    img_scaled = (img - img_min) / (img_max - img_min)

    # current_brightness = np.median(non_zero_luminance)
    current_brightness = np.mean(img_scaled)
    target_brightness=0.35

    if current_brightness < target_brightness:
        # Calculate gamma value
        gamma = np.log(target_brightness) / np.log(current_brightness + 1e-6)
        gamma = max(0.1, min(gamma, 2.0))  # Allow a wider range for gamma
        
        # Apply gamma correction
        np.power(img_scaled, gamma, out=img_scaled)
        np.clip(img_scaled, 0, 1.0, out=img_scaled)

    return img_scaled

def center_crop(img, target_height=2017, target_width=2028):
    """Center crop images to the smallest common dimensions"""
    _, h, w = img.shape
    # If the image is smaller, calculate padding
    pad_h = max(0, target_height - h)
    pad_w = max(0, target_width - w)

    if pad_h > 0 or pad_w > 0:
        # Pad with black pixels (value 0)
        img = np.pad(
            img,
            ((0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
            mode='constant',
            constant_values=0
        )
        h, w = img.shape[1:]  # Update dimensions after padding

    # Calculate cropping indices
    start_h = (h - target_height) // 2
    start_w = (w - target_width) // 2
    return img[:, start_h:start_h+target_height, start_w:start_w+target_width]

def read_img_duo(before_path, after_path, im_name):
    """Read cropped Sentinel-2 image pair and change map."""
    # read images
    before_img_path = os.path.join(before_path, im_name, "files", "composite.tif")
    after_img_path = os.path.join(after_path, im_name, "files", "composite.tif")

    I1 = load_4band_image(before_img_path)
    I2 = load_4band_image(after_img_path)

    return I1, I2

def reshape_for_torch(I):
    """Transpose image for PyTorch coordinates."""
    #     out = np.swapaxes(I,1,2)
    #     out = np.swapaxes(out,0,1)
    #     out = out[np.newaxis,:]
    out = I.transpose((2, 0, 1))
    return from_numpy(out)

def get_diff_features():
    """Get difference features."""
    

class ChangeDetectionDataset(Dataset):
    """Change Detection dataset class, used for both training and test data."""

    def __init__(self, path, before_path=None, after_path=None, diff_features_path=None, transform=None, normalise=True, use_diff_features=False):
        """
        Args:
            path (string): Path to the CSV file with annotations.
            before_path (string): Path to the directory containing 'before' images (optional if using diff features).
            after_path (string): Path to the directory containing 'after' images (optional if using diff features).
            diff_features_path (string): Path to the pre-extracted difference features (optional).
            transform (callable, optional): Optional transform to be applied on a sample.
            normalise (bool): Whether to normalize the images.
            use_diff_features (bool): If True, load pre-extracted difference features instead of images.
        """
        self.normalise = normalise
        self.transform = transform
        self.path = path
        self.before_path = before_path
        self.after_path = after_path
        self.use_diff_features = use_diff_features

        # Read the CSV and extract the timeline_id and binary label columns
        self.df = read_csv(path)
        self.names = self.df["timeline_id"].tolist()
        self.labels = self.df["event"].tolist()
        self.n_imgs = len(self.names)

        # Load pre-extracted difference features if specified
        if self.use_diff_features:
            if diff_features_path is None:
                raise ValueError("diff_features_path must be provided when use_diff_features=True")
            self.diff_features = np.load(diff_features_path)

    def __len__(self):
        return self.n_imgs
    
    def is_valid_idx(self, idx):
        """Check if files for this index exist and are readable."""
        try:
            im_name = self.names[idx]
            before_img_path = os.path.join(self.before_path, im_name, "files", "composite.tif")
            after_img_path = os.path.join(self.after_path, im_name, "files", "composite.tif")
            
            # Check if files exist
            if not os.path.exists(before_img_path) or not os.path.exists(after_img_path):
                print(f"Missing image files for {im_name}")
                return False
                
            return True
        except Exception as e:
            print(f"Error checking index {idx}: {str(e)}")
            return False

    def __getitem__(self, idx):
        if self.use_diff_features:
            # Load pre-extracted difference features
            diff_features = self.diff_features[idx]
            label = self.labels[idx]
            image_id = self.names[idx]

            # Convert to PyTorch tensor
            diff_features = torch.from_numpy(diff_features).float()
            sample = {'diff_features': diff_features, 'label': label, 'image_id': image_id}

        else:
            # Load raw images
            im_name = self.names[idx]
            I1, I2 = read_img_duo(self.before_path, self.after_path, str(im_name))
            label = self.labels[idx]

            if self.normalise:
                mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                I1 = (I1 - mean) / std
                I2 = (I2 - mean) / std

            I1 = center_crop(I1)
            I2 = center_crop(I2)

            # Convert to tensor
            I1 = torch.from_numpy(I1).float().unsqueeze(0)
            I2 = torch.from_numpy(I2).float().unsqueeze(0)

            sample = {'I1': I1, 'I2': I2, 'label': label, 'image_id': im_name}

        if self.transform:
            sample = self.transform(sample)

        return sample