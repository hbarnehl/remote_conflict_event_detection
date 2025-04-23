import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import torch
from torch import nn
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
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt



import time
import datetime

import logging

def evaluate_best_model(model, val_loader, criterion, device):
    """Evaluate the best model and generate visualizations"""
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            diff_feature = batch['diff_features'].squeeze(1).to(device)
            labels = batch['label'].float().to(device).unsqueeze(1)

            outputs = model(before_img=None, after_img=None, diff_features=diff_feature)

            # Apply sigmoid for predictions
            preds = torch.sigmoid(outputs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss /= len(val_loader)
    
    # Convert predictions and labels to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    predictions = (all_preds > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, predictions)
    f1 = f1_score(all_labels, predictions)
    cm = confusion_matrix(all_labels, predictions)
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    
    # Log metrics
    logger.info(f"Best Model Evaluation:")
    logger.info(f"  Val Loss: {val_loss:.4f}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    logger.info(f"  AUC: {roc_auc:.4f}")
    
    # Visualize results
    visualize_results(cm, fpr, tpr, roc_auc)

def visualize_results(cm, fpr, tpr, roc_auc, output_dir="../figures/"):
    """Visualize confusion matrix and ROC curve and save them to a folder"""
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    # Plot ROC curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, 'b-', label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    roc_curve_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_curve_path)  # Save the ROC curve
    plt.close()  # Close the figure to free memory

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['No Change', 'Change'])
    plt.yticks(tick_marks, ['No Change', 'Change'])

    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(confusion_matrix_path)  # Save the confusion matrix
    plt.close()  # Close the figure to free memory

    logger.info(f"Figures saved: {roc_curve_path}, {confusion_matrix_path}")

# def analyze_feature_importance(model, num_features=20):
#     """
#     Analyze which features are most important for the classifier's predictions.
    
#     Args:
#         model: Trained change detection model
#         num_features: Number of top features to display
    
#     Returns:
#         Tuple of (feature_indices, feature_weights) for top positive and negative weights
#     """
#     # Get original model if compiled
#     if hasattr(model, '_orig_mod'):
#         model = model._orig_mod
    
#     # Extract weights based on classifier type
#     if model.classifier_type == 'linear':
#         # For L1RegularizedLinear
#         if hasattr(model.classifier, 'weight'):
#             weights = model.classifier.weight.data.cpu().numpy().flatten()
#         else:
#             # For regular Linear layer
#             weights = model.classifier.linear.weight.data.cpu().numpy().flatten()
#     elif model.classifier_type == 'mlp':
#         # For MLP, extract weights from the first layer
#         weights = model.classifier[0].weight.data.cpu().numpy().flatten()
#     else:
#         logger.warning(f"Unknown classifier type: {model.classifier_type}")
#         return None
    
#     # Get absolute values for overall importance
#     if weights.ndim == 2:  # For MLP, sum weights across neurons
#         abs_weights = np.abs(weights).sum(axis=0)
#     else:  # For linear classifier
#         abs_weights = np.abs(weights)

#     # Sort features by importance
#     sorted_indices = np.argsort(abs_weights)[::-1]  # Descending order
#     top_indices = sorted_indices[:num_features]
#     top_weights = abs_weights[top_indices]
    
#     # Log results
#     logger.info("\n===== Feature Importance Analysis =====")
#     logger.info(f"Total input features: {len(abs_weights)}")
    
#     logger.info("\nTop Positive Features (most important):")
#     for i, (idx, weight) in enumerate(zip(top_indices, top_weights)):
#         logger.info(f"  Feature {idx}: {weight:.6f}")
    
#     # Calculate overall feature importance statistics
#     logger.info("\nFeature Weight Statistics:")
#     logger.info(f"  Mean absolute weight: {abs_weights.mean():.6f}")
#     logger.info(f"  Max absolute weight: {abs_weights.max():.6f}")
#     logger.info(f"  Number of weights > 0.01: {np.sum(abs_weights > 0.01)}")
#     logger.info(f"  Number of weights > 0.001: {np.sum(abs_weights > 0.001)}")
#     logger.info(f"  Weight sparsity: {np.sum(abs_weights < 0.001) / len(weights):.2%}")

#         # Visualize feature importance
#     plt.figure(figsize=(10, 6))
#     plt.barh(range(num_features), top_weights, color=['green' if w > 0 else 'red' for w in top_weights])
#     plt.yticks(range(num_features), [f'Feature {i}' for i in top_indices])
#     plt.xlabel('Weight')
#     plt.ylabel('Feature')
#     plt.title(f'Top {num_features} Feature Weights')
#     plt.tight_layout()
#     plt.show()

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
# logger.info("Starting finetuning process...")
# Set the logging level to INFO 
logger.setLevel(logging.INFO)


def print_time():
    current_time = time.time()
    formatted_time = datetime.datetime.fromtimestamp(current_time).strftime('%Y-%m-%d_%H:%M:%S')
    print(f"{formatted_time}")
    return formatted_time

def print_cuda_memory():
    logger.info(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    logger.info(f"Max allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB\n")


def train(model, train_loader, val_loader, 
        lr=0.001, epochs=10, device='cuda',
        gradient_accumulation_steps=4):
    
    # Binary cross entropy loss
    #pos_weight = torch.tensor([7054 / 1346]).to(device)
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
        # logger.info(f"Epoch {epoch+1}/{epochs}:\n")
        
        # Training phase
        # logger.info("Training...\n")
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()  # Zero gradients at the beginning of epoch
        
        for batch_idx, batch in enumerate(train_loader):
            # logger.info(f"Batch {batch_idx+1}/{len(train_loader)}:")
            # Get inputs
            diff_feature = batch['diff_features'].squeeze(1).to(device)
            labels = batch['label'].float().to(device).unsqueeze(1)

            # Forward pass
            outputs = model(before_img=None, after_img=None, diff_features=diff_feature)
            loss = criterion(outputs, labels)

            # Track the full loss for reporting
            train_loss += loss.item()
            
            # Scale the loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
        
            # Add regularization term if using L1RegularizedLinear
            if hasattr(model.classifier, 'regularization_term'):
                loss = loss + model.classifier.regularization_term / gradient_accumulation_steps
            elif isinstance(model.classifier, nn.Sequential) and hasattr(model.classifier[0], 'regularization_term'):
                # If regularization is in the first layer of sequential
                loss = loss + model.classifier[0].regularization_term / gradient_accumulation_steps
        
            # Backward pass
            loss.backward()
                   
            # Optimizer step every n batches or at the end of epoch
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()

            # if batch_idx % 10 == 0:
            #     logger.info(f"  Batch {batch_idx+1}/{len(train_loader)}")
                
            # Add this in train_sliding_window_model after the first backward pass
            # if epoch == 0 and batch_idx == 0:
                # check_grads_flow(model, "After first backward pass")
            
        train_loss /= len(train_loader)
        
        # logger.info("Validating...\n")

        # Validation phase with memory optimization
        val_loss, accuracy, f1 = validate_model(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step(val_loss)
        
        logger.info(f'Epoch {epoch+1}/{epochs}:\n')
        logger.info(f'  Train Loss: {train_loss:.4f}')
        logger.info(f'  Val Loss: {val_loss:.4f}')
        logger.info(f'  Accuracy: {accuracy:.4f}')
        logger.info(f'  F1 Score: {f1:.4f}\n')

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
        for batch_idx, batch in enumerate(val_loader):

            #logger.info(f"Batch {batch_idx+1}/{len(val_loader)}:")

            diff_feature = batch['diff_features'].squeeze(1).to(device)
            labels = batch['label'].float().to(device).unsqueeze(1)

            outputs = model(before_img=None, after_img=None, diff_features=diff_feature)

            # Apply sigmoid for predictions (but not for loss calculation)
            preds = torch.sigmoid(outputs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Store predictions
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # if batch_idx % 10 == 0:
            #     logger.info(f"  Batch {batch_idx+1}/{len(val_loader)}")
    
    val_loss /= len(val_loader)
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    predictions = (all_preds > 0.5).astype(int)
    accuracy = accuracy_score(all_labels, predictions)
    f1 = f1_score(all_labels, predictions)
    
    return val_loss, accuracy, f1

def finetune_change_detection_model(
    classifier_type="linear",
    l1_lambda=0.2, batch_size=100, epochs_classifier=5,
    dropout1=0.3, dropout2=0.3,
    learning_rate=0.01, gradient_accumulation_steps=4,
    annotations_path="../data/feature_annotations.csv", 
    output_dir="../data/model_results/",
    diff_features_path="../data/diff_features.npy",
):


    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True  # May speed up training
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load MAE model
    # logger.info("Loading ViT model...")
    mae_model = ViT_Win_RVSA(img_size=512)
    checkpoint = torch.load("../data/model_weights/vit-b-checkpoint-1599.pth", map_location='cpu')['model']
    state_dict = mae_model.state_dict()

    for k in ['head.weight', 'head.bias']:
        if k in checkpoint and checkpoint[k].shape != state_dict[k].shape:
            logger.info(f"Removing key {k} from pretrained checkpoint")
            del checkpoint[k]
    

    # interpolate position embedding
    interpolate_pos_embed(mae_model, checkpoint)

    mae_model.load_state_dict(checkpoint, strict=False)

    # 2. Create dataset and dataloaders
    # logger.info("Creating datasets...")
    dataset = ChangeDetectionDataset(
        path=annotations_path,
        use_diff_features=True,
        diff_features_path=diff_features_path
    )

    # Split into train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
############################################ Change num_workers once on server

    # Create dataloaders
    # logger.info("Creating dataloaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4)
    
    # 3. Create end-to-end model
    # logger.info("Creating end-to-end model...")
    model = ChangeDetectionModel(
        feature_extractor=mae_model,
        classifier_type=classifier_type,
        window_size=512,
        overlap=32,
        feature_pooling="max",
        feature_combination="diff_first",
        l1_lambda=l1_lambda,
        freeze_features=True,
        head_only=True
    )

    # Move model to device BEFORE compiling
    model.to(device)

    # 4. Training stages

    # Stage 1: Train only the classifier
    # logger.info("\n=== Stage 1: Training classifier only ===")
    model, val_loss, accuracy, f1 = train(
        model, train_loader, val_loader,
        lr=learning_rate, 
        epochs=epochs_classifier,
        device=device,
        gradient_accumulation_steps=gradient_accumulation_steps
    )

    evaluate_best_model(model, val_loader, criterion=torch.nn.BCEWithLogitsLoss(), device=device)        
    # analyze_feature_importance(model)

    current_time = print_time()

    # Save intermediate model
    # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'sample_size':len(dataset),
    #     'feature_pooling': "max",
    #     'window_size': 512,
    #     'overlap': 32,
    #     'classifier_type': model.classifier_type,
    #     'feature_combination': "diff_frst",
    #     'l1_lambda': l1_lambda,
    #     'val_loss': val_loss,
    #     'accuracy': accuracy,
    #     'f1_score': f1
    # }, os.path.join(output_dir, f'stage1_classifier_only_{current_time}.pth'))

   
    # logger.info("Training complete and model saved!")
    return model, val_loss, accuracy, f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Change Detection Model")
    parser.add_argument("--annotations_path", type=str, default="../data/feature_annotations.csv", help="Path to annotations CSV")
    parser.add_argument("--output_dir", type=str, default="../data/model_results/", help="Directory to save finetuned models")
    parser.add_argument("--l1_lambda", type=float, default=0.2, help="L1 regularization lambda")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for training")
    parser.add_argument("--dropout1", type=float, default=0.2, help="Dropout rate for first layer")
    parser.add_argument("--dropout2", type=float, default=0.2, help="Dropout rate for second layer")
    parser.add_argument("--epochs_classifier", type=int, default=25, help="Number of epochs for classifier training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--classifier_type", type=str, default="mlp", help="Type of classifier ('linear' or 'mlp')")
    parser.add_argument("--diff_features_path", type=str, default="../data/diff_features.npy", help="diff features path")

    args = parser.parse_args()
    
    finetune_change_detection_model(
        annotations_path=args.annotations_path,
        output_dir=args.output_dir,
        l1_lambda=args.l1_lambda,
        batch_size=args.batch_size,
        epochs_classifier=args.epochs_classifier,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        classifier_type=args.classifier_type,
        diff_features_path=args.diff_features_path,
        dropout1=args.dropout1,
        dropout2=args.dropout2
    )