'''
Key Changes and Improvements:
    Memory Efficiency: The code processes one image pair at a time and clears CUDA cache after each processing step to handle your 4GB GPU constraint.

    Pooling Options: I've included multiple pooling methods to experiment with:
        avg: Average pooling of all tokens (except CLS)
        max: Max pooling of tokens
        cls: Using only the CLS token
        all: Concatenating CLS with averaged patch tokens
    
    Sample Limiting: Added a sample_limit parameter to control how many images to process (200 as you specified).

    Visualizations: Added proper evaluation metrics and visualizations including ROC curve, confusion matrix, and feature importance.

    Saving Intermediate Results: The features are saved to avoid recomputation during experimentation
'''


import torch
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import torch.nn.functional as F

# Import functions from your existing module
def process_image_pair_diff_first(row_data, before_dir, after_dir, pooling_method, device):
    """Process a single image pair, first computing differences then pooling"""
    image_id, label = row_data
    # Build paths
    before_path = os.path.join(before_dir, str(image_id) + '.npz')
    after_path = os.path.join(after_dir, str(image_id) + '.npz')
    
    # Skip if files don't exist
    if not os.path.exists(before_path) or not os.path.exists(after_path):
        return None
        
    try:
        # Load features
        before_data = np.load(before_path)
        after_data = np.load(after_path)
        before_features = before_data['features']
        after_features = after_data['features']
        
        # Convert to torch
        before_features_torch = torch.tensor(before_features)
        after_features_torch = torch.tensor(after_features)
        
        # Calculate token-level differences first
        diff_features_torch = after_features_torch - before_features_torch
        
        # Now apply pooling to the difference features
        if pooling_method == 'avg':
            # Average pool all tokens except CLS
            pooled_diff = diff_features_torch[:, 1:, :].mean(dim=1).numpy()
        elif pooling_method == 'max':
            # Max pool all tokens except CLS
            pooled_diff = diff_features_torch[:, 1:, :].max(dim=1)[0].numpy()
        elif pooling_method == 'cls':
            # Use only the CLS token difference
            pooled_diff = diff_features_torch[:, 0, :].numpy()
        elif pooling_method == 'all':
            # Concatenate CLS difference with average of patch token differences
            cls_diff = diff_features_torch[:, 0, :].numpy()
            avg_diff = diff_features_torch[:, 1:, :].mean(dim=1).numpy()
            pooled_diff = np.concatenate([cls_diff, avg_diff], axis=1)
        
        return (pooled_diff.flatten(), label, image_id)
    
    except Exception as e:
        print(f"Error processing image {image_id}: {e}")
        return None

def process_image_pair(row_data, before_dir, after_dir, pooling_method, device):
    """Process a single image pair in parallel"""
    image_id, label = row_data
    
    # Build paths
    before_path = os.path.join(before_dir, str(image_id) + '.npz')
    after_path = os.path.join(after_dir, str(image_id) + '.npz')
    
    # Skip if files don't exist
    if not os.path.exists(before_path) or not os.path.exists(after_path):
        return None
        
    try:
        # Load features (using numpy directly to avoid PyTorch overhead)
        before_data = np.load(before_path)
        after_data = np.load(after_path)
        before_features = before_data['features']
        after_features = after_data['features']
        
        # Convert to torch for pooling
        before_features_torch = torch.tensor(before_features)
        after_features_torch = torch.tensor(after_features)
        
        # Apply pooling
        if pooling_method == 'avg':
            pooled_before = before_features_torch[:, 1:, :].mean(dim=1).numpy()
            pooled_after = after_features_torch[:, 1:, :].mean(dim=1).numpy()
        elif pooling_method == 'max':
            pooled_before = before_features_torch[:, 1:, :].max(dim=1)[0].numpy()
            pooled_after = after_features_torch[:, 1:, :].max(dim=1)[0].numpy()
        elif pooling_method == 'cls':
            pooled_before = before_features_torch[:, 0, :].numpy()
            pooled_after = after_features_torch[:, 0, :].numpy()
        elif pooling_method == 'all':
            cls_before = before_features_torch[:, 0, :].numpy()
            avg_before = before_features_torch[:, 1:, :].mean(dim=1).numpy()
            pooled_before = np.concatenate([cls_before, avg_before], axis=1)
            
            cls_after = after_features_torch[:, 0, :].numpy()
            avg_after = after_features_torch[:, 1:, :].mean(dim=1).numpy()
            pooled_after = np.concatenate([cls_after, avg_after], axis=1)
        
        # Calculate difference
        diff_features = pooled_after - pooled_before
        
        return (diff_features.flatten(), label, image_id)
    
    except Exception as e:
        print(f"Error processing image {image_id}: {e}")
        return None

def create_change_detection_dataset(before_dir, after_dir, label_path, checkpoint_path, 
                                   pooling_method='avg', sample_limit=200, device='cuda'):
    """
    Create a dataset for change detection
    
    Args:
        before_dir: Directory with before images
        after_dir: Directory with after images
        label_path: Path to CSV with labels
        checkpoint_path: Path to model checkpoint
        pooling_method: Method for pooling features
        sample_limit: Maximum number of samples to process
        device: Device to run on
        
    Returns:
        X_diff: Feature differences
        y: Labels
    """
    # Load labels
    labels_df = pd.read_csv(label_path)
    
    # Limit samples if needed
    if sample_limit and sample_limit < len(labels_df):
        labels_df = labels_df.sample(sample_limit, random_state=42)
    
    X_diff = []
    y = []
    image_ids = []
    
    # Process each image pair
    for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
        image_id = row['timeline_id']
        label = row['event']  # Assuming 1=change, 0=no change
        
        # Build paths
        before_path = os.path.join(before_dir, str(image_id)+ '.npz')
        after_path = os.path.join(after_dir, str(image_id) + '.npz')
        
        # Skip if files don't exist
        if not os.path.exists(before_path) or not os.path.exists(after_path):
            print(f"Skipping image {image_id} - files not found")
            continue
            
        try:
            # Extract and pool features
            before_features, after_features = load_features(before_path, after_path, device=device)
            
            pooled_before_features, pooled_after_features = pool_features(before_features, after_features, method=pooling_method)
            
            # Calculate difference
            diff_features = pooled_after_features - pooled_before_features
            
            # Store results
            X_diff.append(diff_features.cpu().numpy())
            y.append(label)
            image_ids.append(image_id)
            
            # Free up CUDA memory
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing image {image_id}: {e}")
    
    # Convert to numpy arrays
    X_diff = np.array(X_diff)
    y = np.array(y)
    
    return X_diff, y, image_ids

def train_lasso_logistic_regression(X_train, y_train, C=0.1):
    """
    Train a Lasso logistic regression model
    
    Args:
        X_train: Training features
        y_train: Training labels
        C: Regularization parameter (1/lambda)
        
    Returns:
        Trained model
    """
    # Create and train model
    model = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        C=C,
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Get predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
    }
    
    # ROC AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    metrics['auc'] = auc(fpr, tpr)
    metrics['fpr'] = fpr
    metrics['tpr'] = tpr
    
    return metrics

def visualize_results(model, metrics, X_test, y_test, image_ids_test=None):
    """
    Visualize model results
    
    Args:
        model: Trained model
        metrics: Metrics dictionary
        X_test: Test features
        y_test: Test labels
        image_ids_test: List of image IDs for the test set
    """
    # 1. Plot ROC curve
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(metrics['fpr'], metrics['tpr'], 'b-', label=f'AUC = {metrics["auc"]:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    
    # 2. Plot confusion matrix
    plt.subplot(1, 2, 2)
    cm = metrics['confusion_matrix']
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
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    plt.tight_layout()
    plt.show()
    
    # 3. Feature importance
    if hasattr(model, 'coef_'):
        top_n = 20
        coef = model.coef_[0]
        
        # Get indices of features with highest absolute weights
        top_indices = np.argsort(np.abs(coef))[-top_n:]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(top_n), coef[top_indices])
        plt.yticks(range(top_n), [f'Feature {i}' for i in top_indices])
        plt.xlabel('Weight')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Weights')
        plt.tight_layout()
        plt.show()

def run_change_detection_demo():
    """Complete change detection demo pipeline"""
    # Paths
    before_dir = '../data/features/before'
    after_dir = '../data/features/after'
    label_path = '../data/changes.csv'
    
    # Parameters
    pooling_method = 'avg'  # 'avg', 'max', 'cls', or 'all'
    sample_limit = 200  # Limit number of samples
    
    # 1. Create dataset
    print("Creating dataset...")
    X_diff, y, image_ids = create_change_detection_dataset(
        before_dir=before_dir,
        after_dir=after_dir,
        label_path=label_path,
        checkpoint_path=checkpoint_path,
        pooling_method=pooling_method,
        sample_limit=sample_limit
    )
    
    print(f"Dataset created with {len(X_diff)} samples")
    print(f"Feature vector shape: {X_diff.shape}")
    
    # 2. Save features to avoid recomputation (optional)
    np.save('change_detection_features.npy', X_diff)
    np.save('change_detection_labels.npy', y)
    np.save('change_detection_image_ids.npy', np.array(image_ids))
    
    # 3. Split data
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X_diff, y, image_ids, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Train model
    print("Training Lasso Logistic Regression model...")
    model = train_lasso_logistic_regression(X_train, y_train, C=0.1)
    
    # 5. Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    
    # 6. Visualize results
    visualize_results(model, metrics, X_test, y_test, ids_test)
    
    return model, X_diff, y, image_ids

# # Run the demo
# if __name__ == "__main__":
#     model, X_diff, y, image_ids = run_change_detection_demo()