import torch.utils.checkpoint as checkpoint
import torch.nn as nn
import torch

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def print_cuda_memory():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB\n")

# this classifier replicates the logistic regression of the demo
class L1RegularizedLinear(nn.Module):
    def __init__(self, in_features, out_features, l1_lambda=0.01):
        """
        Linear layer with built-in L1 regularization (similar to LASSO regression)
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            l1_lambda: L1 regularization strength (similar to 1/C in sklearn)
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.l1_lambda = l1_lambda
        self.regularization_term = 0.0
    
    def forward(self, x):
        out = self.linear(x)
        # Calculate L1 norm of weights (excluding bias)
        l1_term = self.l1_lambda * self.linear.weight.abs().sum()
        # Store regularization term for access during training
        self.regularization_term = l1_term
        return out

class ChangeDetectionModel(nn.Module):
    def __init__(self, feature_extractor, classifier_type='linear', window_size=224, overlap=56, 
             feature_pooling='cls', feature_combination='concatenate', l1_lambda=0.01, freeze_features=False,
             head_only=False, dropout1=0.3, dropout2=0.3):
        """
        End-to-end model for change detection with sliding window feature extraction
        
        Args:
            feature_extractor: Pre-trained feature extraction model (MAE model)
            classifier_type: Type of classifier to use ('linear' or 'mlp')
            window_size: Size of sliding window for feature extraction
            overlap: Overlap between windows
            feature_pooling: Method to pool features ('cls', 'avg', 'max', or 'attention')
            feature_combination: How to combine before/after features ('concatenate', 'difference', 'diff_first')
            l1_lambda: L1 regularization strength for linear classifier (similar to 1/C in sklearn)
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.window_size = window_size
        self.overlap = overlap
        self.feature_pooling = feature_pooling
        self.feature_combination = feature_combination
        self.l1_lambda = l1_lambda
        self.classifier_type = classifier_type
        self.freeze_features = freeze_features
        self.head_only = head_only

        
        # By default, freeze the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # Determine feature dimension by accessing the attribute correctly
        # For MaskedAutoencoderViT the embed_dim is stored as a regular attribute
        if hasattr(self.feature_extractor, 'embed_dim'):
            self.feature_dim = self.feature_extractor.embed_dim
        else:
            # Fallback to accessing via __dict__ or setting manually if needed
            self.feature_dim = getattr(self.feature_extractor, 'embed_dim', 768)
        
        # Create classifier based on feature combination method
        input_dim = self.feature_dim * 2 if feature_combination == 'concatenate' else self.feature_dim
        
        if classifier_type == 'linear':
            # Use regularized linear layer instead of standard linear
            self.classifier = L1RegularizedLinear(input_dim, 1, l1_lambda=l1_lambda)
        elif classifier_type == 'mlp':
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 512),  # Wider first layer
                torch.nn.BatchNorm1d(512),        # Add batch normalization
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout1),            # Slightly less dropout
                torch.nn.Linear(512, 128),        # Add another layer
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout2),
                torch.nn.Linear(128, 1)
            )
        
        # Optional attention layer for feature pooling
        if feature_pooling == 'attention':
            self.attention = torch.nn.Sequential(
                torch.nn.Linear(self.feature_dim, 128),
                torch.nn.Tanh(),
                torch.nn.Linear(128, 1)
            )
    
    def extract_features_sliding_window(self, image, training=False):
        """Extract features using sliding window approach with gradient flow if training"""
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = image.unsqueeze(0)  # [1, C, H, W]
        
        _, C, H, W = image.shape
        stride = self.window_size - self.overlap
        
        # Calculate number of windows
        n_windows_h = max(1, (H - self.window_size + stride) // stride)
        n_windows_w = max(1, (W - self.window_size + stride) // stride)
        
        # Create batches of windows
        window_positions = []
        window_batches = []
        current_batch = []
        
        for h in range(0, H - self.window_size + 1, stride):
            for w in range(0, W - self.window_size + 1, stride):
                patch = image[:, :, h:h+self.window_size, w:w+self.window_size]
                current_batch.append(patch)
                window_positions.append((h, w))
                
                window_batches.append(patch)
                current_batch = []
        
        # Process batches and extract features
        feature_map = {}
        
        if training and not self.freeze_features:
            # During training, process each window individually to maintain proper gradient flow
            for i, window in enumerate(window_batches):
                h, w = window_positions[i]
                
                with torch.set_grad_enabled(True):
                    # Extract intermediate features using hooks instead of forward_encoder
                    features = self.feature_extractor.forward_features(window)
                    feature_map[(h, w)] = features
                
                # Intentionally break references to the input window and extracted features
                del features, window
       
        else:
            # During inference, use torch.no_grad for efficiency
            with torch.no_grad():
                for i, window in enumerate(window_batches):
                    h, w = window_positions[i]

                    # Extract intermediate features
                    features = self.feature_extractor.forward_features(window)
                    feature_map[(h, w)] = features

                # Intentionally break references to the input window and extracted features
                del features, window
        # Merge feature maps
        merged_features = self.merge_feature_map(feature_map, (H, W))

        # Clean up feature map to free memory
        del feature_map

        if self.feature_combination != 'diff_first':
            # Apply feature pooling
            pooled_features = self.pool_features(merged_features)

            # Clean up merged features to free memory
            del merged_features
        
            return pooled_features
        else:
            # If using diff_first, return the full feature map for later processing
            return merged_features
        
    # def get_intermediate_features(self, x):
    #     """Extract intermediate features from the ViT_Win_RVSA model"""
    #     # Get patch embeddings
    #     B, C, H, W = x.shape
    #     x, (Hp, Wp) = self.feature_extractor.patch_embed(x)
        
    #     # Add positional embedding if present
    #     if self.feature_extractor.pos_embed is not None:
    #         x = x + self.feature_extractor.pos_embed[:, 1:]  # Skip cls token position
    #     x = self.feature_extractor.pos_drop(x)
        
    #     # Process through blocks
    #     for i, blk in enumerate(self.feature_extractor.blocks):
    #         x = checkpoint.checkpoint(blk, x, Hp, Wp)
        
    #     # Apply normalization
    #     x = self.feature_extractor.norm(x)
        
    #     # Create a CLS-like token by averaging features (similar to pooling)
    #     cls_token = x.mean(dim=1, keepdim=True)
        
    #     # Concatenate the pooled token as a CLS token
    #     features = torch.cat([cls_token, x], dim=1)
        
    #     return features
    
    def merge_feature_map(self, feature_map, image_shape):
        """Merge overlapping feature patches without circular region filtering"""
        H, W = image_shape
        patch_size = 16  # ViT-B patch size
        
        # Calculate feature dimensions
        feature_h = H // patch_size
        feature_w = W // patch_size
        
        # Get device and dimensions from first feature
        first_feature = next(iter(feature_map.values()))
        device = first_feature.device
        batch_size = first_feature.size(0)
        feature_dim = first_feature.shape[-1]
        
        # Pre-allocate memory only once
        merged = torch.zeros((batch_size, feature_h, feature_w, feature_dim), device=device)
        counts = torch.ones((batch_size, feature_h, feature_w, 1), device=device) * 1e-8  # Avoid div by zero
        
        # Collect CLS tokens more efficiently
        cls_tokens = torch.zeros((len(feature_map), batch_size, feature_dim), device=device)
        
        # Process all windows at once
        for idx, ((h, w), feat) in enumerate(feature_map.items()):
            # Store CLS token without creating new tensors
            cls_tokens[idx] = feat[:, 0, :]
            
            # Calculate token positions
            h_token = h // patch_size
            w_token = w // patch_size
            tokens_per_side = self.window_size // patch_size
            
            # Calculate boundaries
            h_end = min(h_token + tokens_per_side, feature_h)
            w_end = min(w_token + tokens_per_side, feature_w)
            h_size = h_end - h_token
            w_size = w_end - w_token
            
            # Get patch tokens efficiently (avoid unnecessary copying)
            feat_reshaped = feat[:, 1:, :].reshape(batch_size, tokens_per_side, tokens_per_side, feature_dim)
            
            # Add features in-place
            merged[:, h_token:h_end, w_token:w_end] += feat_reshaped[:, :h_size, :w_size]
            counts[:, h_token:h_end, w_token:w_end] += 1
        
        # Normalize with single division
        merged.div_(counts)
        
        # Compute mean CLS token efficiently
        avg_cls_token = cls_tokens.mean(dim=0, keepdim=True).view(batch_size, 1, feature_dim)
        
        # Reshape merged features once
        merged_flat = merged.reshape(batch_size, feature_h*feature_w, feature_dim)
        
        # Concatenate efficiently
        return torch.cat([avg_cls_token, merged_flat], dim=1)
    
    def pool_features(self, features):
        """Pool features based on selected method"""
        if self.feature_pooling == 'cls':
            # Use only CLS token
            return features[:, 0, :]
        
        elif self.feature_pooling == 'avg':
            # Average all patch tokens (excluding CLS token)
            return torch.mean(features[:, 1:, :], dim=1)
        
        elif self.feature_pooling == 'max':
            # Max pooling of patch tokens
            return torch.max(features[:, 1:, :], dim=1)[0]
        
        elif self.feature_pooling == 'attention':
            # Attention-weighted pooling
            attention_weights = self.attention(features[:, 1:, :])  # [B, L, 1]
            attention_weights = torch.softmax(attention_weights, dim=1)  # [B, L, 1]
            pooled = torch.sum(features[:, 1:, :] * attention_weights, dim=1)  # [B, D]
            return pooled
        
        else:
            # Default: concatenate CLS with averaged patch tokens
            cls_token = features[:, 0, :]
            avg_patch = torch.mean(features[:, 1:, :], dim=1)
            return torch.cat([cls_token, avg_patch], dim=1)
    
    def forward(self, before_img=None, after_img=None, diff_features=None):
        """Forward pass through the model with memory optimization for circular images"""
        
        # Set training mode
        is_training = self.training

        if self.head_only:
            logits = self.classifier(diff_features)
            return logits

        if self.feature_combination == 'diff_first':
            # Conditionally use torch.no_grad() for feature extraction
            before_features_full = self.extract_features_sliding_window(before_img, training=is_training)
            after_features_full = self.extract_features_sliding_window(after_img, training=is_training)

            # Calculate difference between token representations
            diff_features =  after_features_full - before_features_full          
            # Free memory
            del before_features_full, after_features_full
            torch.cuda.empty_cache()

            # Apply pooling
            pooled_diff = self.pool_features(diff_features)

            # Free memory
            del diff_features
            torch.cuda.empty_cache()

            logits = self.classifier(pooled_diff)

            # Free memory
            del pooled_diff
            torch.cuda.empty_cache()

        elif self.feature_combination == 'difference':
           # Extract and pool features separately, then take difference
            before_features = self.extract_features_sliding_window(before_img, training=is_training)
            after_features = self.extract_features_sliding_window(after_img, training=is_training)

            # Calculate difference between pooled features
            diff_features = after_features - before_features
            
            # Free memory
            del before_features, after_features  # Fixed variable names
            torch.cuda.empty_cache()
            
            # Classify based on the difference
            logits = self.classifier(diff_features)
            
        else:  # 'concatenate' (default)
           # Extract features from both images
            before_features = self.extract_features_sliding_window(before_img, training=is_training)
            after_features = self.extract_features_sliding_window(after_img, training=is_training)

            # Combine features by concatenation
            combined_features = torch.cat([before_features, after_features], dim=1)
            
            # Free memory
            del before_features, after_features  # Fixed variable names
            torch.cuda.empty_cache()
            
            # Classify
            logits = self.classifier(combined_features)
        
        return logits