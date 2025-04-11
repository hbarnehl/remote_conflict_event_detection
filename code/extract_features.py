import torch
import numpy as np
import os
from tqdm import tqdm
import argparse
from pathlib import Path
import sys
import rasterio

script_dir = '../../Remote-Sensing-RVSA/'
sys.path.append(script_dir)

# Import your custom modules
from MAEPretrain_SceneClassification.models_mae import mae_vit_base_patch16_dec512d8b

from torch_helpers import ChangeDetectionDataset

def load_model(checkpoint_path, device='cuda'):
    """
    Load the pretrained model from checkpoint
    
    Args:
        checkpoint_path: Path to the model checkpoint
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    model = mae_vit_base_patch16_dec512d8b()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)
    model.eval()
    return model

def load_4band_image(path):
    """Load a 4-band image and ensure it has dimensions [1, channels, height, width]"""
    with rasterio.open(path) as src:
        # Read the bands (assuming band 1 is infrared, band 2 is red, band 3 is green, band 4 is blue)
        blue = src.read(1)
        green = src.read(2)
        red = src.read(3)

        # Stack the bands into a single array
        img = np.stack((red, green, blue), axis=0)  # [channels, height, width]

        # Normalize the image
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img = (img / 255.0 - mean) / std

        # Convert to tensor and add batch dimension
        img = torch.from_numpy(img).float()
        img = img.unsqueeze(0)  # [1, channels, height, width]

    return img


def extract_features(model, image, device='cuda', batch_size=1):
    """
    Extract features from an image using the pretrained model
    
    Args:
        model: Pretrained model
        image: Input tensor of shape [C, H, W]
        device: Device to run inference on
        batch_size: Batch size for processing
        
    Returns:
        Extracted features
    """
    # Ensure image has batch dimension
    if len(image.shape) == 3:
        image = image.unsqueeze(0)  # [1, C, H, W]
        
    # Move to device
    image = image.to(device)
    
    # Extract features without computing gradients
    with torch.no_grad():
        features, _, _ = model.forward_encoder(image, mask_ratio=0.0)
    
    return features

def process_large_image_efficiently(model, image, window_size=224, overlap=56, device='cuda', batch_size=4):
    """
    Process large images using sliding windows with batched processing
    
    Args:
        model: Pretrained model
        image: Input tensor of shape [C, H, W] or [1, C, H, W]
        window_size: Size of sliding window
        overlap: Overlap between windows
        device: Device to run inference on
        batch_size: Batch size for processing windows
        
    Returns:
        Dictionary mapping window positions to features
    """
    # Ensure image has batch dimension
    if len(image.shape) == 3:
        image = image.unsqueeze(0)  # [1, C, H, W]
    
    _, C, H, W = image.shape
    stride = window_size - overlap
    
    # Calculate number of windows
    n_windows_h = max(1, (H - window_size + stride) // stride)
    n_windows_w = max(1, (W - window_size + stride) // stride)
    total_windows = n_windows_h * n_windows_w
    
    # Create batches of windows for efficient processing
    window_positions = []
    window_batches = []
    current_batch = []
    
    for h in range(0, H - window_size + 1, stride):
        for w in range(0, W - window_size + 1, stride):
            patch = image[:, :, h:h+window_size, w:w+window_size]
            current_batch.append(patch)
            window_positions.append((h, w))
            
            # Process batch when it reaches the desired size
            if len(current_batch) == batch_size:
                window_batches.append(torch.cat(current_batch, dim=0))
                current_batch = []
    
    # Handle the last batch if it's not full
    if current_batch:
        window_batches.append(torch.cat(current_batch, dim=0))
    
    # Process batches and extract features
    feature_map = {}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(window_batches):
            # Process batch
            batch = batch.to(device)
            features, _, _ = model.forward_encoder(batch, mask_ratio=0.0)
            
            # Assign features to their respective positions
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(window_positions))
            
            for i, position_idx in enumerate(range(start_idx, end_idx)):
                h, w = window_positions[position_idx]
                feature_map[(h, w)] = features[i:i+1]  # Keep batch dimension
            
            # Free up memory
            del features
            torch.cuda.empty_cache()
    
    return feature_map

def merge_feature_map(feature_map, image_shape, window_size=224, overlap=56):
    """
    Merge overlapping feature patches while preserving token structure
    
    Args:
        feature_map: Dictionary mapping positions to features
        image_shape: Original image shape (H, W)
        window_size: Window size used for extraction
        overlap: Overlap used between windows
        
    Returns:
        Merged features tensor with CLS token preserved at position 0
    """
    H, W = image_shape
    patch_size = 16  # ViT-B uses 16×16 patches
    
    # Calculate the feature map dimensions (downsampled by patch_size)
    feature_h = H // patch_size
    feature_w = W // patch_size
    feature_dim = next(iter(feature_map.values())).shape[-1]
    
    # Initialize the merged feature map and a count map for averaging
    merged = torch.zeros((1, feature_h, feature_w, feature_dim), device=next(iter(feature_map.values())).device)
    counts = torch.zeros((1, feature_h, feature_w, 1), device=next(iter(feature_map.values())).device)
    
    # Collect all CLS tokens
    cls_tokens = []
    
    # Calculate stride in token space
    stride = (window_size - overlap) // patch_size
    
    # Add each feature patch to the appropriate position
    for (h, w), feat in feature_map.items():
        # Save the CLS token (first token)
        cls_tokens.append(feat[:, 0, :])
        
        # Calculate token position
        h_token = h // patch_size
        w_token = w // patch_size
        
        # Get the actual expected number of tokens based on the window size
        tokens_per_side = window_size // patch_size
        
        # Process patch tokens (skip CLS token)
        feature_tokens = feat[:, 1:, :]
        B, L, D = feature_tokens.shape
        
        # Check if L matches expected token count
        expected_L = tokens_per_side * tokens_per_side
        if L != expected_L:
            print(f"Warning: Feature shape mismatch. Expected {expected_L}, got {L}")
            continue
            
        feat_reshaped = feature_tokens.reshape(B, tokens_per_side, tokens_per_side, D)
        
        # Calculate boundaries making sure we don't go out of bounds
        h_end = min(h_token + tokens_per_side, feature_h)
        w_end = min(w_token + tokens_per_side, feature_w)
        
        # Size of patch that fits in the feature map
        h_size = h_end - h_token
        w_size = w_end - w_token
        
        merged[:, h_token:h_end, w_token:w_end] += feat_reshaped[:, :h_size, :w_size]
        counts[:, h_token:h_end, w_token:w_end] += 1
    
    # Average overlapping regions (avoid division by zero)
    merged = merged / (counts + 1e-8)
    
    # Average all collected CLS tokens
    avg_cls_token = torch.mean(torch.cat(cls_tokens, dim=0), dim=0, keepdim=True).unsqueeze(0)
    
    # Reshape merged features from [B, H, W, D] to [B, L, D] format
    B, H, W, D = merged.shape
    merged_flat = merged.reshape(B, H*W, D)
    
    # Concatenate CLS token at position 0
    merged_with_cls = torch.cat([avg_cls_token, merged_flat], dim=1)
    
    return merged_with_cls


# Function to get a sample by timeline_id
def get_sample_by_timeline_id(dataset, timeline_id):
    # Find the index of the timeline_id in the dataset
    try:
        idx = dataset.names.index(timeline_id)
    except ValueError:
        raise ValueError(f"timeline_id {timeline_id} not found in the dataset.")
    
    # Use __getitem__ to retrieve the sample
    sample = dataset[idx]
    return sample

def extract_and_save_features(image_id, before_dir, after_dir, 
                              output_dir, model, device='cuda', use_merged=True):
    """
    Extract features for a single image pair and save them
    
    Args:
        image_id: ID of the image to process
        before_dir: Directory containing 'before' images
        after_dir: Directory containing 'after' images
        output_dir: Directory to save features
        device: Device to run on
        use_merged: Whether to use merged features or raw feature maps
        
    Returns:
        Paths to the saved feature files
    """
    # initialise dataset
    dataset = ChangeDetectionDataset(path="../data/annotations_ukraine.csv",
                                     before_path=before_dir,
                                     after_path=after_dir,
                                     stride=1,
                                     transform=None,
                                     normalise=True)

    # Create output directories
    before_out_dir = os.path.join(output_dir, 'before')
    after_out_dir = os.path.join(output_dir, 'after')
    Path(before_out_dir).mkdir(exist_ok=True, parents=True)
    Path(after_out_dir).mkdir(exist_ok=True, parents=True)
    
    # Build paths
    before_path = os.path.join(before_dir, str(image_id), 'files', 'composite.tif')
    after_path = os.path.join(after_dir, str(image_id), 'files', 'composite.tif')
    
    # Check if files exist
    if not os.path.exists(before_path) or not os.path.exists(after_path):
        print(f"Error: Files for image {image_id} not found")
        return False  # Indicate failure
    
    # Load images
    sample = get_sample_by_timeline_id(dataset, image_id)
    before_tensor, after_tensor, _ = sample["I1"], sample["I2"], sample["label"]
    
    # Extract features
    try:
        # Process before image
        before_feature_map = process_large_image_efficiently(
            model, before_tensor, window_size=224, overlap=56, device=device, batch_size=4
        )
        
        # Process after image
        after_feature_map = process_large_image_efficiently(
            model, after_tensor, window_size=224, overlap=56, device=device, batch_size=4
        )
        
        # Decide whether to use raw feature maps or merged features
        if use_merged:
            before_features = merge_feature_map(before_feature_map, before_tensor.shape[2:], 
                                              window_size=224, overlap=56)
            after_features = merge_feature_map(after_feature_map, after_tensor.shape[2:], 
                                             window_size=224, overlap=56)
            
            # Convert to CPU and numpy
            before_features = before_features.cpu().numpy()
            after_features = after_features.cpu().numpy()
        else:
            # Convert feature maps to CPU and numpy
            before_features = {pos: feat.cpu().numpy() for pos, feat in before_feature_map.items()}
            after_features = {pos: feat.cpu().numpy() for pos, feat in after_feature_map.items()}
        
        # Save features
        before_output_path = os.path.join(before_out_dir, f"{image_id}.npz")
        after_output_path = os.path.join(after_out_dir, f"{image_id}.npz")
        
        if use_merged:
            np.savez_compressed(before_output_path, features=before_features)
            np.savez_compressed(after_output_path, features=after_features)
        else:
            # Convert dictionary to lists for saving
            positions = list(before_features.keys())
            before_values = [before_features[pos] for pos in positions]
            after_values = [after_features[pos] for pos in positions]
            
            # Save as numpy arrays
            np.savez_compressed(before_output_path, 
                               positions=positions, 
                               features=before_values)
            np.savez_compressed(after_output_path, 
                              positions=positions, 
                              features=after_values)
        
        return True  # Indicate success
        
    except Exception as e:
        print(f"Error processing image {image_id}: {e}")
        return False  # Indicate failure
    finally:
        # Clean up
        torch.cuda.empty_cache()

def batch_extract_features(image_ids, before_dir, after_dir, checkpoint_path, 
                          output_dir, device='cuda', use_merged=True, skip_existing=True):
    """
    Extract features for multiple image pairs
    
    Args:
        image_ids: list of event ids to process
        before_dir: Directory containing 'before' images
        after_dir: Directory containing 'after' images
        checkpoint_path: Path to model checkpoint
        output_dir: Directory to save features
        device: Device to run on
        use_merged: Whether to use merged features or raw feature maps
    """
    print("Loading model...")
    model = load_model(checkpoint_path, device)

    # Process each image pair
    success_count = 0

    if skip_existing:
        # exclude image ids that already exist in the output directory
        before_out_dir = os.path.join(output_dir, 'before')
        after_out_dir = os.path.join(output_dir, 'after')
        existing_before = [int(os.path.splitext(f)[0]) for f in os.listdir(before_out_dir)]
        existing_after = [int(os.path.splitext(f)[0]) for f in os.listdir(after_out_dir)]
        old_ids = image_ids.copy()
        image_ids = [img_id for img_id in image_ids if img_id not in existing_before and img_id not in existing_after]
        print(f"Skipping {len(old_ids) - len(image_ids)} existing features.")
    
    for image_id in tqdm(image_ids, desc="Extracting features"):         
        success = extract_and_save_features(
            image_id, before_dir, after_dir, output_dir, 
            model, device=device, use_merged=use_merged
        )
        
        if success:
            success_count += 1
    
    print(f"Successfully extracted features for {success_count} out of {len(image_ids)} image pairs.")

def load_features(image_id, features_dir, subset='before', use_merged=True):
    """
    Load extracted features for a specific image
    
    Args:
        image_id: ID of the image
        features_dir: Base directory containing features
        subset: 'before' or 'after'
        use_merged: Whether the features are merged or raw feature maps
        
    Returns:
        Loaded features
    """
    file_path = os.path.join(features_dir, subset, f"{image_id}.npz")
    
    if not os.path.exists(file_path):
        print(f"Error: Features for image {image_id} not found")
        return None
    
    data = np.load(file_path, allow_pickle=True)
    
    if use_merged:
        return data['features']
    else:
        # Reconstruct dictionary
        positions = data['positions']
        features = data['features']
        
        return {pos: feat for pos, feat in zip(positions, features)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features for change detection")
    parser.add_argument("--image_ids", type=list, required=True, help="list of image pair ids")
    parser.add_argument("--before_dir", type=str, required=True, help="Directory with 'before' images")
    parser.add_argument("--after_dir", type=str, required=True, help="Directory with 'after' images")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save features")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--use_merged", action="store_true", help="Use merged features instead of raw feature maps")
    
    args = parser.parse_args()
    
    batch_extract_features(
        args.image_ids, 
        args.before_dir, 
        args.after_dir, 
        args.checkpoint, 
        args.output_dir,
        args.device,
        args.use_merged
    )