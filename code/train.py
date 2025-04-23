import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import torch
from torch import nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from vitae_models.models_mae import mae_vit_base_patch16_dec512d8b
from vitae_models.vit_win_rvsa import ViT_Win_RVSA
from util.pos_embed import interpolate_pos_embed
from cd_dataset import ChangeDetectionDataset
from change_detection_model import ChangeDetectionModel
import argparse
from psutil import cpu_count

import time
import datetime

import logging

def analyze_feature_importance(model, num_features=20):
    """
    Analyze which features are most important for the classifier's predictions.
    
    Args:
        model: Trained change detection model
        num_features: Number of top features to display
    
    Returns:
        Tuple of (feature_indices, feature_weights) for top positive and negative weights
    """
    # Get original model if compiled
    if hasattr(model, '_orig_mod'):
        model = model._orig_mod
    
    # Extract weights based on classifier type
    if model.classifier_type == 'linear':
        # For L1RegularizedLinear
        if hasattr(model.classifier, 'weight'):
            weights = model.classifier.weight.data.cpu().numpy().flatten()
        else:
            # For regular Linear layer
            weights = model.classifier.linear.weight.data.cpu().numpy().flatten()
    elif model.classifier_type == 'mlp':
        # For MLP, extract weights from the first layer
        weights = model.classifier[0].weight.data.cpu().numpy().flatten()
    else:
        logger.warning(f"Unknown classifier type: {model.classifier_type}")
        return None
    
    # Get absolute values for overall importance
    abs_weights = np.abs(weights)
    
    # Find indices of top positive and top negative weights
    sorted_indices = np.argsort(abs_weights)[::-1]  # Sort by absolute value (descending)
    
    # Get top features
    top_indices = sorted_indices[:num_features]
    top_weights = weights[top_indices]
    
    # Separate positive and negative weights
    pos_mask = top_weights > 0
    neg_mask = top_weights < 0
    
    pos_indices = top_indices[pos_mask]
    pos_weights = top_weights[pos_mask]
    
    neg_indices = top_indices[neg_mask]
    neg_weights = top_weights[neg_mask]
    
    # Log results
    logger.info("\n===== Feature Importance Analysis =====")
    logger.info(f"Total features: {len(weights)}")
    
    logger.info("\nTop Positive Features (changes that suggest damage):")
    for i, (idx, weight) in enumerate(zip(pos_indices, pos_weights)):
        if i < 10:  # Print top 10
            logger.info(f"  Feature {idx}: {weight:.6f}")
    
    logger.info("\nTop Negative Features (changes that suggest no damage):")
    for i, (idx, weight) in enumerate(zip(neg_indices, neg_weights)):
        if i < 10:  # Print top 10
            logger.info(f"  Feature {idx}: {weight:.6f}")
    
    # Calculate overall feature importance statistics
    logger.info("\nFeature Weight Statistics:")
    logger.info(f"  Mean absolute weight: {abs_weights.mean():.6f}")
    logger.info(f"  Max absolute weight: {abs_weights.max():.6f}")
    logger.info(f"  Number of weights > 0.01: {np.sum(abs_weights > 0.01)}")
    logger.info(f"  Number of weights > 0.001: {np.sum(abs_weights > 0.001)}")
    logger.info(f"  Weight sparsity: {np.sum(abs_weights < 0.001) / len(weights):.2%}")
    
    return (pos_indices, pos_weights), (neg_indices, neg_weights)

def check_grads_flow(model, stage="After unfreezing"):
    """Check if gradients are flowing into feature extractor parameters"""
    has_grad = {}
    no_grad = {}
    
    # Check which parameters require gradients
    for name, param in model.named_parameters():
        if 'feature_extractor' in name:
            if param.requires_grad:
                has_grad[name] = param.shape
            else:
                no_grad[name] = param.shape
    
    logger.info(f"\n{stage}:")
    logger.info(f"  Feature extractor parameters with gradients: {len(has_grad)}")
    logger.info(f"  Feature extractor parameters without gradients: {len(no_grad)}")
    
    # Log some examples of parameters with grads
    if has_grad:
        examples = list(has_grad.keys())[:3]
        logger.info(f"  Examples of parameters with gradients: {examples}")

def track_parameter_changes(model, stage_name="initial"):
    """Store parameter snapshots to track changes during training"""
    if not hasattr(model, 'param_snapshots'):
        model.param_snapshots = {}
    
    # Track a few important layers (last block attention and norm layers)
    tracked_layers = [
        'feature_extractor.blocks.11.attn.proj.weight',
        'feature_extractor.blocks.10.attn.proj.weight',
        'feature_extractor.norm.weight'
    ]
    
    snapshot = {}
    for name, param in model.named_parameters():
        if name in tracked_layers:
            snapshot[name] = param.detach().cpu().clone()
    
    model.param_snapshots[stage_name] = snapshot
    return model

# Add this to print the changes at the end
def print_parameter_changes(model):
    """Print changes in tracked parameters"""
    if not hasattr(model, 'param_snapshots') or len(model.param_snapshots) < 2:
        logger.info("Not enough parameter snapshots to compare")
        return
    
    logger.info("\n===== Parameter Change Analysis =====")
    
    for name in model.param_snapshots["before_finetuning"]:
        before = model.param_snapshots["before_finetuning"][name]
        after = model.param_snapshots["after_finetuning"][name]
        
        abs_diff = (after - before).abs()
        mean_change = abs_diff.mean().item()
        max_change = abs_diff.max().item()
        
        logger.info(f"Layer: {name}")
        logger.info(f"  Mean absolute change: {mean_change:.6f}")
        logger.info(f"  Max absolute change: {max_change:.6f}")
        logger.info(f"  Change percentage: {100 * mean_change / before.abs().mean().item():.2f}%")


# initialise logger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(levelname)s-%(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("Starting finetuning process...")
# Set the logging level to INFO 
logger.setLevel(logging.INFO)


def print_time():
    current_time = time.time()
    formatted_time = datetime.datetime.fromtimestamp(current_time).strftime('%Y-%m-%d_%H:%M')
    print(f"{formatted_time}")
    return formatted_time

def print_cuda_memory():
    logger.info(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    logger.info(f"Max allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB\n")


def train_sliding_window_model(model, train_loader, val_loader, 
                              lr=0.001, epochs=10, device='cuda',
                              gradient_accumulation_steps=4):
    """
    Train the sliding window model with gradient accumulation for memory efficiency
    
    Args:
        model: ChangeDetectionModel
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        lr: Learning rate
        epochs: Number of training epochs
        device: Device to train on
        gradient_accumulation_steps: Number of steps to accumulate gradient before optimizer step
    """
       
    # Initialize scaler for AMP
    scaler = GradScaler("cuda")

    # Binary cross entropy loss
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Optimizer with different learning rates - use orig_model for parameter access
    optimizer = torch.optim.Adam([
        {'params': model.classifier.parameters(), 'lr': lr},
        # Use lower learning rate for feature extractor when unfreezing
        {'params': model.feature_extractor.parameters(), 'lr': lr * 0.1}
    ])
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}:\n")
        
        # Training phase
        logger.info("Training...\n")
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()  # Zero gradients at the beginning of epoch
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')):
            logger.info(f"Batch {batch_idx+1}/{len(train_loader)}:")
            # Get inputs
            before_imgs = batch['I1'].squeeze(1).to(device)
            after_imgs = batch['I2'].squeeze(1).to(device)
            labels = batch['label'].float().to(device).unsqueeze(1)

            # Use autocast to enable mixed precision
            with autocast("cuda"):
                outputs = model(before_imgs, after_imgs)
                loss = criterion(outputs, labels)

                # Track the full loss for reporting (not scaled by gradient_accumulation_steps)
                train_loss += loss.item()
                
                # Scale the loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
            
                # Add regularization term if using L1RegularizedLinear
                if hasattr(model.classifier, 'regularization_term'):
                    loss = loss + model.classifier.regularization_term / gradient_accumulation_steps
                elif isinstance(model.classifier, nn.Sequential) and hasattr(model.classifier[0], 'regularization_term'):
                    # If regularization is in the first layer of sequential
                    loss = loss + model.classifier[0].regularization_term / gradient_accumulation_steps
            

            # Scale loss and do backward pass
            scaler.scale(loss).backward()
                       
            # Optimizer step every n batches or at the end of epoch
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Unscale before clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step with scaler
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Clear CUDA cache periodically
                torch.cuda.empty_cache()
            # Add this in train_sliding_window_model after the first backward pass
            if epoch == 0 and batch_idx == 0:
                check_grads_flow(model, "After first backward pass")
        
        del loss, before_imgs, after_imgs, labels
            
        train_loss /= len(train_loader)

        print_cuda_memory()
        torch.cuda.reset_peak_memory_stats()
        
        logger.info("Validating...\n")

        # Validation phase with memory optimization
        val_loss, accuracy, f1 = validate_model(model, val_loader, criterion, device)
        
        print_cuda_memory()
        torch.cuda.reset_peak_memory_stats()

        # Update learning rate
        scheduler.step(val_loss)
        
        
        logger.info(f'Epoch {epoch+1}/{epochs}:\n')
        logger.info(f'  Train Loss: {train_loss:.4f}')
        logger.info(f'  Val Loss: {val_loss:.4f}')
        logger.info(f'  Accuracy: {accuracy:.4f}')
        logger.info(f'  F1 Score: {f1:.4f}\n')

        analyze_feature_importance(model)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            
    # Load best model
    model.load_state_dict(best_model_state)
    return model, val_loss, accuracy, f1

def validate_model(model, val_loader, criterion, device):
    """Validate model with memory optimization"""
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Process in smaller batches to save memory
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc='Validation')):

            logger.info(f"Batch {batch_idx+1}/{len(val_loader)}:")

            before_imgs = batch['I1'].squeeze(1).to(device)
            after_imgs = batch['I2'].squeeze(1).to(device)
            labels = batch['label'].float().to(device).unsqueeze(1)

            outputs = model(before_imgs, after_imgs)

            # Apply sigmoid for predictions (but not for loss calculation)
            preds = torch.sigmoid(outputs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Store predictions
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss /= len(val_loader)
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    predictions = (all_preds > 0.5).astype(int)
    accuracy = accuracy_score(all_labels, predictions)
    f1 = f1_score(all_labels, predictions)

    del before_imgs, after_imgs, labels
    
    return val_loss, accuracy, f1

def unfreeze_feature_extractor_progressively(model, strategy='last_blocks', n_blocks=2):
    """
    Unfreeze parts of the feature extractor for finetuning
    
    Args:
        model: ChangeDetectionModel
        strategy: Unfreezing strategy ('last_blocks', 'all', or 'none')
        n_blocks: Number of blocks to unfreeze if strategy is 'last_blocks'
    """
    # First freeze all parameters in feature extractor
    for param in model.feature_extractor.parameters():
        param.requires_grad = False
    
    if strategy == 'none':
        return model
    
    elif strategy == 'all':
        # Unfreeze all parameters
        for param in model.feature_extractor.parameters():
            param.requires_grad = True
    
    elif strategy == 'last_blocks':
        # Unfreeze only the last n blocks
        total_blocks = len(model.feature_extractor.blocks)
        start_idx = max(0, total_blocks - n_blocks)
        
        # Unfreeze last n blocks
        for i in range(start_idx, total_blocks):
            for param in model.feature_extractor.blocks[i].parameters():
                param.requires_grad = True
        
        # Also unfreeze norm layer
        for param in model.feature_extractor.norm.parameters():
            param.requires_grad = True
    
    # Print trainable parameters info
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
    
    return model

def finetune_change_detection_model(
    window_size=244, overlap=56, classifier_type="linear",
    feature_pooling="max", feature_combination="diff_first", l1_lambda=0.2,
    batch_size=100, epochs_classifier=5, epochs_finetuning=3,
    learning_rate=0.01, gradient_accumulation_steps=4,
    from_finetuning="false",
    before_path="../data/images_ukraine_extracted_before/",
    after_path="../data/images_ukraine_extracted_after/",
    annotations_path="../data/annotations_ukraine.csv", 
    checkpoint_path="../data/model_weights/vit-b-checkpoint-1599.pth",
    output_dir="../data/model_weights/"
):
    """
    Main function for end-to-end finetuning of a change detection model.
    This function performs the following steps:
    1. Loads a pre-trained MAE (Masked Autoencoder) model.
    2. Creates a dataset and splits it into training and validation sets.
    3. Constructs an end-to-end change detection model.
    4. Trains the model in two stages:
       - Stage 1: Trains only the classifier.
       - Stage 2: Finetunes the last few blocks of the feature extractor.
    Returns:
        ChangeDetectionModel: The trained change detection model.
    """

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True  # May speed up training
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load MAE model
    logger.info("Loading ViT model...")
    mae_model = ViT_Win_RVSA(img_size=window_size)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')['model']
    state_dict = mae_model.state_dict()

    for k in ['head.weight', 'head.bias']:
        if k in checkpoint and checkpoint[k].shape != state_dict[k].shape:
            logger.info(f"Removing key {k} from pretrained checkpoint")
            del checkpoint[k]
    

    # interpolate position embedding
    interpolate_pos_embed(mae_model, checkpoint)

    mae_model.load_state_dict(checkpoint, strict=False)

    # 2. Create dataset and dataloaders
    logger.info("Creating datasets...")
    dataset = ChangeDetectionDataset(
        path=annotations_path,
        before_path=before_path,
        after_path=after_path,
        normalise=True
    )

    # Split into train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
############################################ Change num_workers once on server

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4)
    
    # 3. Create end-to-end model
    logger.info("Creating end-to-end model...")
    model = ChangeDetectionModel(
        feature_extractor=mae_model,
        classifier_type=classifier_type,
        window_size=window_size,
        overlap=overlap,
        feature_pooling=feature_pooling,
        feature_combination=feature_combination,
        l1_lambda=l1_lambda,
        freeze_features=True
    )

    # Move model to device BEFORE compiling
    model.to(device)

    # 4. Training stages
    
    if from_finetuning !="true":
        # Stage 1: Train only the classifier
        logger.info("\n=== Stage 1: Training classifier only ===")
        model, val_loss, accuracy, f1 = train_sliding_window_model(
            model, train_loader, val_loader,
            lr=learning_rate, 
            epochs=epochs_classifier,
            device=device,
            gradient_accumulation_steps=gradient_accumulation_steps
        )        

        current_time = print_time()

        # Save intermediate model
        torch.save({
            'model_state_dict': model.state_dict(),
            'sample_size':len(dataset),
            'feature_pooling': feature_pooling,
            'window_size': window_size,
            'overlap': overlap,
            'classifier_type': model.classifier_type,
            'feature_combination': feature_combination,
            'l1_lambda': l1_lambda,
            'val_loss': val_loss,
            'accuracy': accuracy,
            'f1_score': f1
        }, os.path.join(output_dir, f'stage1_classifier_only_{current_time}.pth'))
    
    # Stage 2: Finetune last few blocks of feature extractor
    logger.info("\n=== Stage 2: Finetuning last blocks ===")
    model.freeze_features = False  # Enable gradients for feature extractor
    model = unfreeze_feature_extractor_progressively(model, strategy='last_blocks', n_blocks=2)
    check_grads_flow(model, "After unfreezing")
    model = track_parameter_changes(model, "before_finetuning")
        
    
    model, val_loss, accuracy, f1 = train_sliding_window_model(
        model, train_loader, val_loader,
        lr=learning_rate * 0.1,  # Lower learning rate for finetuning
        epochs=epochs_finetuning,
        device=device,
        gradient_accumulation_steps=gradient_accumulation_steps
    )
    model = track_parameter_changes(model, "after_finetuning")
    print_parameter_changes(model)

    current_time = print_time()

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'sample_size':len(dataset),
        'feature_pooling': feature_pooling,
        'window_size': window_size,
        'overlap': overlap,
        'classifier_type': model.classifier_type,
        'feature_combination': feature_combination,
        'l1_lambda': l1_lambda,
        'val_loss': val_loss,
        'accuracy': accuracy,
        'f1_score': f1
    }, os.path.join(output_dir, f'finetuned_change_detection_model_{current_time}.pth'))
    
    logger.info("Training complete and model saved!")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Change Detection Model")
    parser.add_argument("--before_path", type=str, default="../data/images_ukraine_extracted_before/", help="Path to 'before' images")
    parser.add_argument("--after_path", type=str, default="../data/images_ukraine_extracted_after/", help="Path to 'after' images")
    parser.add_argument("--annotations_path", type=str, default="../data/annotations_ukraine.csv", help="Path to annotations CSV")
    parser.add_argument("--checkpoint_path", type=str, default="../data/model_weights/vit-b-checkpoint-1599.pth", help="Path to pretrained MAE checkpoint")
    parser.add_argument("--output_dir", type=str, default="../data/model_weights/", help="Directory to save finetuned models")
    parser.add_argument("--window_size", type=int, default=224, help="Sliding window size")
    parser.add_argument("--overlap", type=int, default=56, help="Sliding window overlap")
    parser.add_argument("--feature_pooling", type=str, default="max", choices=["cls", "avg", "max", "attention"], help="Feature pooling method")
    parser.add_argument("--feature_combination", type=str, default="diff_first", choices=["concatenate", "difference", "diff_first"], help="Feature combination method")
    parser.add_argument("--l1_lambda", type=float, default=0.2, help="L1 regularization lambda")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for training")
    parser.add_argument("--epochs_classifier", type=int, default=5, help="Number of epochs for classifier training")
    parser.add_argument("--epochs_finetuning", type=int, default=3, help="Number of epochs for finetuning")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--classifier_type", type=str, default="linear", help="Type of classifier ('linear' or 'mlp')")
    parser.add_argument("--from_finetuning", type=str, default="false", help="if training should go directly to finetuning")    

    args = parser.parse_args()
    
    finetune_change_detection_model(
        before_path=args.before_path,
        after_path=args.after_path,
        annotations_path=args.annotations_path,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        window_size=args.window_size,
        overlap=args.overlap,
        feature_pooling=args.feature_pooling,
        feature_combination=args.feature_combination,
        l1_lambda=args.l1_lambda,
        batch_size=args.batch_size,
        epochs_classifier=args.epochs_classifier,
        epochs_finetuning=args.epochs_finetuning,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        classifier_type=args.classifier_type,
        from_finetuning=args.from_finetuning
    )