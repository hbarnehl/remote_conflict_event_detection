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

def print_cuda_memory():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB\n")


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

    model.to(device)
    
    print("after loading model onto device:")
    print_cuda_memory()

    # Binary cross entropy loss
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Optimizer with different learning rates
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
        print("at epoch beginning:")
        print_cuda_memory()
        # Training phase
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()  # Zero gradients at the beginning of epoch
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')):
            # Get inputs
            before_imgs = batch['I1'].squeeze(1).to(device)
            after_imgs = batch['I2'].squeeze(1).to(device)
            labels = batch['label'].float().to(device).unsqueeze(1)
            print("after loading images and labels to device:")
            print_cuda_memory()
            
            # Use autocast to enable mixed precision
            with autocast("cuda"):
                outputs = model(before_imgs, after_imgs)
                loss = criterion(outputs, labels) / gradient_accumulation_steps
            
                # Add regularization term if using L1RegularizedLinear
                if hasattr(model.classifier, 'regularization_term'):
                    loss = loss + model.classifier.regularization_term / gradient_accumulation_steps
                elif isinstance(model.classifier, nn.Sequential) and hasattr(model.classifier[0], 'regularization_term'):
                    # If regularization is in the first layer of sequential
                    loss = loss + model.classifier[0].regularization_term / gradient_accumulation_steps
                
            # Scale loss and do backward pass
            scaler.scale(loss).backward()
            print("after backward:")
            print_cuda_memory()
            
                       
            # Optimizer step every n batches or at the end of epoch
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Unscale before clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step with scaler
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                print("after optimiser step:")
                print_cuda_memory()
                
                # Clear CUDA cache periodically
                if (batch_idx + 1) % (gradient_accumulation_steps * 5) == 0:
                    torch.cuda.empty_cache()
        
        del loss, before_imgs, after_imgs, labels
            
        train_loss /= len(train_loader)
        torch.cuda.empty_cache()

        print("after training:")
        print_cuda_memory()
        
        # Validation phase with memory optimization
        val_loss, accuracy, f1 = validate_model(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Accuracy: {accuracy:.4f}')
        print(f'  F1 Score: {f1:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            
    # Load best model
    model.load_state_dict(best_model_state)
    return model

def validate_model(model, val_loader, criterion, device):
    """Validate model with memory optimization"""
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Process in smaller batches to save memory
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            before_imgs = batch['I1'].squeeze(1).to(device)
            after_imgs = batch['I2'].squeeze(1).to(device)
            labels = batch['label'].float().to(device).unsqueeze(1)
            print("after loading images and labels to device:")
            print_cuda_memory()
            
            # Process one image at a time if needed
            batch_size = before_imgs.size(0)
            outputs = []
            
            for i in range(batch_size):
                output = model(before_imgs[i:i+1], after_imgs[i:i+1])
                outputs.append(output)
                
                # Clear cache after each image if memory is tight
                torch.cuda.empty_cache()
            
            # Concatenate outputs
            outputs = torch.cat(outputs, dim=0)

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

def unfreeze_feature_extractor_progressively(model, strategy='last_blocks', n_blocks=1):
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
            
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
    
    return model

def finetune_change_detection_model(
    window_size=244, overlap=56, classifier_type="linear", feature_pooling="max", feature_combination="diff_first", l1_lambda=0.2,
    batch_size=100, epochs_classifier=5, epochs_finetuning=3, learning_rate=0.01, gradient_accumulation_steps=4,
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
    Args:
        before_path (str): Path to the "before" images used for change detection.
        after_path (str): Path to the "after" images used for change detection.
        annotations_path (str): Path to the annotations file containing labels for the dataset.
        checkpoint_path (str): Path to the pre-trained MAE model checkpoint.
        output_dir (str): Directory where the trained models and intermediate results will be saved.
        window_size (int): Size of the sliding window used for processing images.
        overlap (float): Overlap ratio between sliding windows (value between 0 and 1).
        feature_pooling: Method to pool features ('cls', 'avg', 'max', or 'attention')
        feature_combination: How to combine before/after features ('concatenate', 'difference', 'diff_first')
        l1_lambda (float): Regularization parameter for L1 loss applied to the classifier.
        batch_size (int): Number of samples per batch during training.
        epochs (int): Total number of epochs for training.
        learning_rate (float): Initial learning rate for the optimizer.
        gradient_accumulation_steps (int): Number of steps for gradient accumulation before updating weights.
        classifier_type (str): Type of classifier to use ('linear' or 'mlp').
    Returns:
        ChangeDetectionModel: The trained change detection model.
    """

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True  # May speed up training
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Before anything:")
    print_cuda_memory()

    # 1. Load MAE model
    print("Loading ViT model...")
    mae_model = ViT_Win_RVSA(img_size=window_size)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')['model']
    state_dict = mae_model.state_dict()

    for k in ['head.weight', 'head.bias']:
        if k in checkpoint and checkpoint[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint[k]
    

    # interpolate position embedding
    interpolate_pos_embed(mae_model, checkpoint)

    mae_model.load_state_dict(checkpoint, strict=False)

    # Modify the model to ensure compatibility with your change detection pipeline
    # 1. Remove the classification head
    if hasattr(mae_model, 'head'):
        mae_model.head = nn.Identity()

    # 2. Create dataset and dataloaders
    print("Creating datasets...")
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_count(logical=False))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=cpu_count(logical=False))
    
    # 3. Create end-to-end model
    print("Creating end-to-end model...")
    model = ChangeDetectionModel(
        feature_extractor=mae_model,
        classifier_type=classifier_type,
        window_size=window_size,
        overlap=overlap,
        feature_pooling=feature_pooling,
        feature_combination=feature_combination,
        l1_lambda=l1_lambda,
    )
    
    print("after creating MAE model:")
    print_cuda_memory()

    # 4. Training stages
    
    # Stage 1: Train only the classifier
    print("\n=== Stage 1: Training classifier only ===")
    model = train_sliding_window_model(
        model, train_loader, val_loader,
        lr=learning_rate, 
        epochs=epochs_classifier,
        device=device,
        gradient_accumulation_steps=gradient_accumulation_steps
    )
    
    # Add at the beginning of each training epoch
    torch.cuda.reset_peak_memory_stats()

    # Save intermediate model
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_pooling': feature_pooling,
        'window_size': window_size,
        'overlap': overlap,
        'classifier_type': model.classifier_type,
        'feature_combination': feature_combination,
        'l1_lambda': l1_lambda
    }, os.path.join(output_dir, 'stage1_classifier_only.pth'))
    
    # Stage 2: Finetune last few blocks of feature extractor
    print("\n=== Stage 2: Finetuning last blocks ===")
    model = unfreeze_feature_extractor_progressively(model, strategy='last_blocks', n_blocks=1)
    model = train_sliding_window_model(
        model, train_loader, val_loader,
        lr=learning_rate * 0.1,  # Lower learning rate for finetuning
        epochs=epochs_finetuning,
        device=device,
        gradient_accumulation_steps=gradient_accumulation_steps
    )
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_pooling': feature_pooling,
        'window_size': window_size,
        'overlap': overlap
    }, os.path.join(output_dir, 'finetuned_change_detection_model.pth'))
    
    print("Training complete and model saved!")
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
        classifier_type=args.classifier_type
    )