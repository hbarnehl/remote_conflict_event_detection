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
        # Read the bands (assuming band 1 is infrared, band 2 is red, band 3 is green, band 4 is blue)
        blue = src.read(1)
        green = src.read(2)
        red = src.read(3)

        # Stack the bands into a single array
        img = np.stack((red, green, blue), axis=0)  # [channels, height, width]

        img = enhance_dark_image(img)

    return img

# Apply adaptive brightness enhancement
def enhance_dark_image(img, target_brightness=0.35):
    
    # Normalize the image data to 0-1 range for display
    img_min, img_max = img.min(), img.max()
    img_scaled = (img - img_min) / (img_max - img_min)


    # luminance = 0.2126 * img_scaled[0, :, :] + 0.7152 * img_scaled[1, :, :] + 0.0722 * img_scaled[2, :, :]

    # # Calculate brightness using the mean of luminance
    # non_zero_luminance = luminance[luminance > 0]

    # current_brightness = np.median(non_zero_luminance)
    current_brightness = np.mean(img_scaled)

    # print(f"Current brightness: {current_brightness:.4f}")
    
    if current_brightness < target_brightness:
        # Calculate gamma value
        gamma = np.log(target_brightness) / np.log(current_brightness + 1e-6)
        gamma = max(0.1, min(gamma, 2.0))  # Allow a wider range for gamma
        
        # Apply gamma correction
        enhanced = np.power(img_scaled, gamma)
        enhanced = np.clip(enhanced, 0, 1.0)
        # print(f"Applied gamma correction: {gamma:.2f}")
        return enhanced
    else:
        return img_scaled

def center_crop(img, target_height=2017, target_width=2028):
    """Center crop images to the smallest common dimensions"""
    _, h, w = img.shape
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



class ChangeDetectionDataset(Dataset):
    """Change Detection dataset class, used for both training and test data."""

    def __init__(self, path, before_path, after_path, stride=None, transform=None, normalise=True):
        """
        Args:
            path (string): Path to the csv file with annotations.
            before_path (string): Path to the directory containing 'before' images.
            after_path (string): Path to the directory containing 'after' images.
            stride (int): Stride for sampling the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.normalise = normalise
        self.transform = transform
        self.path = path
        self.before_path = before_path
        self.after_path = after_path
        self.stride = stride if stride else 1

        # Read the CSV and extract the timeline_id and binary label columns
        self.df = read_csv(path)
        self.names = self.df["timeline_id"].tolist()
        self.labels = self.df["event"].tolist()
        self.n_imgs = len(self.names)

    def __len__(self):
        return self.n_imgs

    def __getitem__(self, idx):
        im_name = self.names[idx]
        I1, I2 = read_img_duo(self.before_path, self.after_path, str(im_name))
        label = self.labels[idx]


        # normalisation to 0-1 range
        # I1 = I1.numpy()
        # I2 = I2.numpy()  #12,128,128

        if self.normalise:
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            I1 = (I1 - mean) / std
            I2 = (I2 - mean) / std


            # NOTE: Both z-score and min-max normalizations are applied sequentially
            # to match the pre-trained model's expected input distribution.

            # kid1 = (I1 - I1.min(axis=(1, 2), keepdims=True))
            # mom1 = (I1.max(axis=(1, 2), keepdims=True) - I1.min(axis=(1, 2), keepdims=True)+ 1e-8)
            # I1 = kid1 / (mom1)

            # kid2 = (I2 - I2.min(axis=(1, 2), keepdims=True))
            # mom2 = (I2.max(axis=(1, 2), keepdims=True) - I2.min(axis=(1, 2), keepdims=True)+ 1e-8)
            # I2 = kid2 / (mom2)

        I1 = center_crop(I1)
        I2 = center_crop(I2)

        # I1 = from_numpy(I1.transpose((0, 1, 2)))
        # I2 = from_numpy(I2.transpose((0, 1, 2)))

        # Convert to tensor and add batch dimension
        I1 = torch.from_numpy(I1).float().unsqueeze(0)
        I2 = torch.from_numpy(I2).float().unsqueeze(0)

        sample = {'I1': I1, 'I2': I2, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample