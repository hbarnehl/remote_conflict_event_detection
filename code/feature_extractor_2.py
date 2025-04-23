import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import torch
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from vitae_models.vit_win_rvsa import ViT_Win_RVSA
from util.pos_embed import interpolate_pos_embed
from cd_dataset import ChangeDetectionDataset
from change_detection_model import ChangeDetectionModel
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(levelname)s-%(message)s',
    handlers=[
        logging.FileHandler("feature_extraction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def find_latest_batch(output_dir):
    """Find the latest batch number in the output directory"""
    if not os.path.exists(output_dir):
        return 0
        
    batch_dirs = [d for d in os.listdir(output_dir) if d.startswith('batch_')]
    if not batch_dirs:
        return 0
        
    # Extract batch numbers from directory names
    batch_numbers = []
    for d in batch_dirs:
        try:
            batch_num = int(d.split('_')[1])
            batch_numbers.append(batch_num)
        except (IndexError, ValueError):
            continue
            
    return max(batch_numbers) if batch_numbers else 0

def print_cuda_memory():
    logger.info(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    logger.info(f"Max allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB\n")

class FeatureExtractor:
    def __init__(
        self,
        window_size=512,
        overlap=32,
        feature_pooling="max",
        feature_combination="diff_first",
        checkpoint_path="../data/model_weights/vit-b-checkpoint-1599.pth",
        device=None,
        last_batch=0
    ):
        """Initialize the feature extractor with the pretrained model"""
        self.window_size = window_size
        self.overlap = overlap
        self.feature_pooling = feature_pooling
        self.feature_combination = feature_combination
        self.last_batch = last_batch
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Load model
        logger.info("Loading ViT model...")
        self.mae_model = ViT_Win_RVSA(img_size=window_size)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')['model']
        state_dict = self.mae_model.state_dict()
        
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint and checkpoint[k].shape != state_dict[k].shape:
                logger.info(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
                
        # Interpolate position embedding
        interpolate_pos_embed(self.mae_model, checkpoint)
        self.mae_model.load_state_dict(checkpoint, strict=False)
        
        # Create feature extractor model
        self.model = ChangeDetectionModel(
            feature_extractor=self.mae_model,
            classifier_type="linear",  # Doesn't matter for feature extraction
            window_size=window_size,
            overlap=overlap,
            feature_pooling=feature_pooling,
            feature_combination=feature_combination,
            freeze_features=True
        )
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
    def extract_features_from_dataloader(self, dataloader, output_dir):
        """Extract features from a dataloader and save to disk"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        feature_data = {
            'diff_features': [],
            'labels': [],
            'image_ids': []
        }

        # Track failed samples
        failed_samples = []
            
        # Process batches
        logger.info("Extracting features...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc='Extracting Features')):
                batch_idx += self.last_batch
                try:
                    before_imgs = batch['I1'].squeeze(1).to(self.device)
                    if batch_idx == 0:
                        logger.info(f"Before images loaded")
                    after_imgs = batch['I2'].squeeze(1).to(self.device)
                    if batch_idx == 0:
                        logger.info(f"After images loaded")
                    labels = batch['label'].numpy()
                    image_ids = batch['image_id'] 
                    if batch_idx == 0:
                        logger.info(f"Image IDs loaded")
                        
                    # Extract features using the sliding window approach
                    before_features = self.model.extract_features_sliding_window(before_imgs, training=False)
                    after_features = self.model.extract_features_sliding_window(after_imgs, training=False)
                    
                    # calculate difference features
                    diff_features = before_features - after_features
                    
                    diff_features = self.model.pool_features(after_features).cpu().numpy()
                    
                    # Store features
                    feature_data['diff_features'].append(diff_features)
                    feature_data['labels'].append(labels)
                    feature_data['image_ids'].extend(image_ids)
                    
                except Exception as e:
                    # Log the error and continue with the next batch
                    logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                    logger.error(f"Affected image IDs: {batch['image_id']}")
                    failed_samples.extend(batch['image_id'])
                    continue
                
                
                # Free up memory
                torch.cuda.empty_cache()
                
                # Report progress every few batches
                if (batch_idx + 1) % 5 == 0:
                    logger.info(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
                
                # after first batch, print CUDA memory usage
                if batch_idx < 3:
                    print_cuda_memory()
                
                # Save features every 10 batches
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                    logger.info(f"Saving features at batch {batch_idx + 1}")
                    for key in ['diff_features', 'labels']:
                        if feature_data[key]:
                            feature_data[key] = np.concatenate(feature_data[key], axis=0)
                    
                    # Save features to disk
                    batch_output_dir = os.path.join(output_dir, f"batch_{batch_idx + 1}")
                    os.makedirs(batch_output_dir, exist_ok=True)
                    np.save(os.path.join(batch_output_dir, 'diff_features.npy'), feature_data['diff_features'])
                    np.save(os.path.join(batch_output_dir, 'labels.npy'), feature_data['labels'])
                    with open(os.path.join(batch_output_dir, 'image_ids.txt'), 'w') as f:
                        for img_id in feature_data['image_ids']:
                            f.write(f"{img_id}\n")
                    
                    # Clear stored features to save memory
                    feature_data['diff_features'] = []
                    feature_data['labels'] = []
                    feature_data['image_ids'] = []
        logger.info("Feature extraction complete!")
        return feature_data
            
def main():
    parser = argparse.ArgumentParser(description="Extract features from images using pretrained model")
    parser.add_argument("--before_path", type=str, default="../data/images_ukraine_extracted_before/", help="Path to 'before' images")
    parser.add_argument("--after_path", type=str, default="../data/images_ukraine_extracted_after/", help="Path to 'after' images")
    parser.add_argument("--annotations_path", type=str, default="../data/annotations_ukraine.csv", help="Path to annotations CSV")
    parser.add_argument("--checkpoint_path", type=str, default="../data/model_weights/vit-b-checkpoint-1599.pth", help="Path to pretrained checkpoint")
    parser.add_argument("--output_dir", type=str, default="../data/features/", help="Directory to save features")
    parser.add_argument("--window_size", type=int, default=512, help="Sliding window size")
    parser.add_argument("--overlap", type=int, default=32, help="Sliding window overlap")
    parser.add_argument("--feature_pooling", type=str, default="max", choices=["cls", "avg", "max", "attention"], help="Feature pooling method")
    parser.add_argument("--feature_combination", type=str, default="diff_first", choices=["concatenate", "difference", "diff_first"], help="Feature combination method")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for feature extraction")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    
    args = parser.parse_args()
    
    # Create output directory with subfolder based on parameters
    output_subfolder = f"w{args.window_size}_o{args.overlap}_{args.feature_pooling}_{args.feature_combination}"
    output_dir = os.path.join(args.output_dir, output_subfolder)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Find the latest batch if resuming
    start_batch = 0
    latest_batch = find_latest_batch(output_dir)
    start_batch = latest_batch
    logger.info(f"Resuming from batch {start_batch + 1}")
    
    # Create dataset and dataloader
    logger.info("Creating dataset...")
    dataset = ChangeDetectionDataset(
        path=args.annotations_path,
        before_path=args.before_path,
        after_path=args.after_path,
        normalise=True
    )
    
    # # Pre-check all files and get valid indices
    # logger.info("Pre-checking files...")
    # valid_indices = []
    # for i in tqdm(range(len(dataset)), desc="Checking files"):
    #     if dataset.is_valid_idx(i):
    #         valid_indices.append(i)

    # logger.info(f"Found {len(valid_indices)}/{len(dataset)} valid samples")
    
    # # Create a subset dataset with only valid indices
    # subset_dataset = torch.utils.data.Subset(dataset, valid_indices)
    # # save the valid indices to a file
    # with open(os.path.join(output_dir, 'valid_indices.txt'), 'w') as f:
    #     for idx in valid_indices:
    #         f.write(f"{idx}\n")

    # # load the valid indices from a file
    # with open(os.path.join(output_dir, 'valid_indices.txt'), 'r') as f:
    #     valid_indices = [int(line.strip()) for line in f.readlines()]

    # # Create a subset dataset with only valid indices
    # subset_dataset = torch.utils.data.Subset(dataset, valid_indices)

    logger.info(f"Dataset size: {len(dataset)}")

    # If resuming, create a subset dataset starting from the next batch
    if start_batch > 0:
        start_idx = start_batch * args.batch_size
        if start_idx >= len(dataset):
            logger.info("All batches already processed. Nothing to do.")
            return
        
        # Create subset starting from the first unprocessed sample
        dataset = torch.utils.data.Subset(dataset, list(range(start_idx, len(dataset))))
        logger.info(f"Starting from sample index {start_idx}, {len(dataset)} samples remaining")
    
    # Create dataloader (no need to shuffle for feature extraction)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    
    
    # Initialize feature extractor
    extractor = FeatureExtractor(
        window_size=args.window_size,
        overlap=args.overlap,
        feature_pooling=args.feature_pooling,
        feature_combination=args.feature_combination,
        checkpoint_path=args.checkpoint_path,
        last_batch=latest_batch
    )
    
    # Extract features
    extractor.extract_features_from_dataloader(dataloader, output_dir)
    
if __name__ == "__main__":
    main()