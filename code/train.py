import torch
from torch import nn
import tqdm
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from vitae_models.models_mae import mae_vit_base_patch16_dec512d8b
from cd_dataset import ChangeDetectionDataset
from change_detection_model import ChangeDetectionModel
import argparse


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
    model.to(device)
    
    # Binary cross entropy loss
    criterion = torch.nn.BCELoss()
    
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
        # Training phase
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()  # Zero gradients at the beginning of epoch
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')):
            # Get inputs
            before_imgs = batch['I1'].squeeze(1).to(device)
            after_imgs = batch['I2'].squeeze(1).to(device)
            labels = batch['label'].float().to(device).unsqueeze(1)
            
            # Forward pass
            outputs = model(before_imgs, after_imgs)
            
            # Calculate loss 
            loss = criterion(outputs, labels) / gradient_accumulation_steps
            
            # Add regularization term if using L1RegularizedLinear
            if hasattr(model.classifier, 'regularization_term'):
                loss = loss + model.classifier.regularization_term / gradient_accumulation_steps
            elif isinstance(model.classifier, nn.Sequential) and hasattr(model.classifier[0], 'regularization_term'):
                # If regularization is in the first layer of sequential
                loss = loss + model.classifier[0].regularization_term / gradient_accumulation_steps
            
            # Backward pass (accumulate gradients)
            loss.backward()
            
            # Scale loss for reporting
            train_loss += loss.item() * gradient_accumulation_steps
            
            # Optimizer step every n batches or at the end of epoch
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()
                
                # Clear CUDA cache periodically
                if (batch_idx + 1) % (gradient_accumulation_steps * 5) == 0:
                    torch.cuda.empty_cache()
            
        train_loss /= len(train_loader)
        
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
            
            # Calculate loss
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Store predictions
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss /= len(val_loader)
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    predictions = (all_preds > 0.5).astype(int)
    accuracy = accuracy_score(all_labels, predictions)
    f1 = f1_score(all_labels, predictions)
    
    return val_loss, accuracy, f1

def unfreeze_feature_extractor_progressively(model, strategy='last_blocks', n_blocks=2):
    """
    Unfreeze parts of the feature extractor for finetuning
    
    Args:
        model: ChangeDetectionModel
        strategy: Unfreezing strategy ('last_blocks', 'all', or 'none')
        n_blocks: Number of blocks to unfreeze if strategy is 'last_blocks'
        
    Returns:
        Updated model with selected parameters unfrozen
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
    
    print("Model finetuning settings:")
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
    before_path, after_path, annotations_path, checkpoint_path, output_dir,
    window_size, overlap, feature_pooling, feature_combination, l1_lambda,
    batch_size, epochs, learning_rate, gradient_accumulation_steps
):
    """Main function for end-to-end finetuning of change detection model"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True  # May speed up training
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load MAE model
    print("Loading MAE model...")
    mae_model = mae_vit_base_patch16_dec512d8b()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    mae_model.load_state_dict(checkpoint['model'], strict=False)
    
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
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 3. Create end-to-end model
    print("Creating end-to-end model...")
    model = ChangeDetectionModel(
        feature_extractor=mae_model,
        classifier_type='mlp',
        window_size=window_size,
        overlap=overlap,
        feature_pooling=feature_pooling,
        feature_combination=feature_combination,
        l1_lambda=l1_lambda,
    )
    
    # 4. Training stages
    
    # Stage 1: Train only the classifier
    print("\n=== Stage 1: Training classifier only ===")
    model = train_sliding_window_model(
        model, train_loader, val_loader,
        lr=learning_rate, 
        epochs=5,
        device=device,
        gradient_accumulation_steps=gradient_accumulation_steps
    )
    
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
    model = unfreeze_feature_extractor_progressively(model, strategy='last_blocks', n_blocks=2)
    model = train_sliding_window_model(
        model, train_loader, val_loader,
        lr=learning_rate * 0.1,  # Lower learning rate for finetuning
        epochs=3,
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
    parser.add_argument("--before_path", type=str, required=True, help="Path to 'before' images")
    parser.add_argument("--after_path", type=str, required=True, help="Path to 'after' images")
    parser.add_argument("--annotations_path", type=str, required=True, help="Path to annotations CSV")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to pretrained MAE checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save finetuned models")
    parser.add_argument("--window_size", type=int, default=224, help="Sliding window size")
    parser.add_argument("--overlap", type=int, default=56, help="Sliding window overlap")
    parser.add_argument("--feature_pooling", type=str, default="max", choices=["cls", "avg", "max", "attention"], help="Feature pooling method")
    parser.add_argument("--feature_combination", type=str, default="diff_first", choices=["concatenate", "difference", "diff_first"], help="Feature combination method")
    parser.add_argument("--l1_lambda", type=float, default=0.0, help="L1 regularization lambda")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    
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
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )