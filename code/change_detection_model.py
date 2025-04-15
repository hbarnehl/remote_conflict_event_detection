import torch.utils.checkpoint as checkpoint
import torch.nn as nn
import torch

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
             feature_pooling='cls', feature_combination='concatenate', l1_lambda=0.01):
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
                torch.nn.Linear(input_dim, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(256, 1)
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
                
                # Process as individual windows to maintain proper gradient flow during training
                if training:
                    window_batches.append(patch)
                    current_batch = []

        # For inference, we batch windows for efficiency
        if not training and current_batch:
            window_batches.append(torch.cat(current_batch, dim=0))
        
        # Process batches and extract features
        feature_map = {}
        
        if training:
            # During training, process each window individually to maintain proper gradient flow
            for i, window in enumerate(window_batches):
                h, w = window_positions[i]
                
                with torch.set_grad_enabled(True):
                    # Extract intermediate features using hooks instead of forward_encoder
                    features = self.get_intermediate_features(window)
                    feature_map[(h, w)] = features
                
                # Intentionally break references to the input window and extracted features
                del features, window
                
                # Force memory reclamation if needed (every N windows)
                if i % 10 == 0 and i > 0:
                    torch.cuda.empty_cache()
                    print(f"After processing window {i+1}/{len(window_batches)}:")
                    print_cuda_memory()
        
        else:
            # During inference, use torch.no_grad for efficiency
            with torch.no_grad():
                for batch_idx, batch in enumerate(window_batches):
                    # Extract intermediate features
                    features = self.get_intermediate_features(batch)
                    
                    # Assign features to their respective positions
                    start_idx = batch_idx * batch.size(0)
                    end_idx = min(start_idx + batch.size(0), len(window_positions))
                    
                    for i, position_idx in enumerate(range(start_idx, end_idx)):
                        h, w = window_positions[position_idx]
                        feature_map[(h, w)] = features[i:i+1]  # Keep batch dimension
        print("After extracting features:")
        print_cuda_memory()

        # Merge feature maps
        merged_features = self.merge_feature_map(feature_map, (H, W))
        print(f"After merging feature map:")
        print_cuda_memory()


        # Clean up feature map to free memory
        del feature_map
        torch.cuda.empty_cache()

        if self.feature_combination != 'diff_first':
            # Apply feature pooling
            pooled_features = self.pool_features(merged_features)

            # Clean up merged features to free memory
            del merged_features
            torch.cuda.empty_cache()
        
            return pooled_features
        else:
            # If using diff_first, return the full feature map for later processing
            return merged_features
        
    def get_intermediate_features(self, x):
        """Extract intermediate features from the ViT_Win_RVSA model"""
        # Get patch embeddings
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.feature_extractor.patch_embed(x)
        
        # Add positional embedding if present
        if self.feature_extractor.pos_embed is not None:
            x = x + self.feature_extractor.pos_embed[:, 1:]  # Skip cls token position
        x = self.feature_extractor.pos_drop(x)
        
        # Process through blocks
        for i, blk in enumerate(self.feature_extractor.blocks):
            x = checkpoint.checkpoint(blk, x, Hp, Wp)
        
        # Apply normalization
        x = self.feature_extractor.norm(x)
        
        # Create a CLS-like token by averaging features (similar to pooling)
        cls_token = x.mean(dim=1, keepdim=True)
        
        # Concatenate the pooled token as a CLS token
        features = torch.cat([cls_token, x], dim=1)
        
        return features
    
    def merge_feature_map(self, feature_map, image_shape):
        """Merge overlapping feature patches with circular region optimization"""
        H, W = image_shape
        patch_size = 16  # ViT-B patch size
        
        # Calculate feature dimensions
        feature_h = H // patch_size
        feature_w = W // patch_size
        feature_dim = next(iter(feature_map.values())).shape[-1]
        
        # Get device
        first_feature = next(iter(feature_map.values()))
        device = first_feature.device
        batch_size = first_feature.size(0)  # Get actual batch size
        
        # Create circle mask
        h_center, w_center = H/2, W/2
        radius = min(H, W)/2 - 1
        
        # Calculate center of each patch in image space
        y_centers = torch.arange(feature_h, device=device) * patch_size + patch_size/2
        x_centers = torch.arange(feature_w, device=device) * patch_size + patch_size/2
        
        # Create patch distance matrix
        y_grid, x_grid = torch.meshgrid(y_centers, x_centers)
        patch_distances = torch.sqrt((y_grid - h_center)**2 + (x_grid - w_center)**2)
        
        # Create circular mask (1 for patches in circle, 0 outside)
        circle_mask = (patch_distances <= (radius + patch_size/2)).float().unsqueeze(0).unsqueeze(-1)
        # Expand circle_mask to match batch size
        circle_mask = circle_mask.expand(batch_size, -1, -1, -1)
        
        # Initialize sparse accumulators with proper batch size
        merged = torch.zeros((batch_size, feature_h, feature_w, feature_dim), device=device)
        counts = torch.zeros((batch_size, feature_h, feature_w, 1), device=device)
        
        # Get indices of patches within the circle (as a mask)
        valid_patches = (circle_mask > 0).squeeze(-1)  # [1, H, W]
        
        # Only process patches that overlap with the circle
        cls_tokens = []
        
        # Add each feature patch - only to relevant locations
        for (h, w), feat in feature_map.items():
            # Save CLS token
            cls_tokens.append(feat[:, 0, :])
            
            # Calculate token position
            h_token = h // patch_size
            w_token = w // patch_size
            tokens_per_side = self.window_size // patch_size
            
            # Process patch tokens (skip CLS token)
            feature_tokens = feat[:, 1:, :]
            B, L, D = feature_tokens.shape
            
            # Reshape to spatial grid
            feat_reshaped = feature_tokens.reshape(B, tokens_per_side, tokens_per_side, D)
            
            # Calculate boundaries
            h_end = min(h_token + tokens_per_side, feature_h)
            w_end = min(w_token + tokens_per_side, feature_w)
            h_size = h_end - h_token
            w_size = w_end - w_token
            
            # Check if this window overlaps with the circular region at all
            window_valid_mask = valid_patches[:, h_token:h_end, w_token:w_end]
            if not window_valid_mask.any():
                continue  # Skip this window entirely if outside circle
                
            # Add features only within circle - apply mask
            window_mask = circle_mask[:, h_token:h_end, w_token:w_end, :]
            merged[:, h_token:h_end, w_token:w_end] += feat_reshaped[:, :h_size, :w_size] * window_mask
            counts[:, h_token:h_end, w_token:w_end] += window_mask
        
        # Average overlapping regions (only where counts > 0)
        merged = torch.where(counts > 0, merged / (counts + 1e-8), merged)
        
        # Average all CLS tokens
        avg_cls_token = torch.mean(torch.cat(cls_tokens, dim=0), dim=0, keepdim=True)  # Shape: [1, D]
        # Expand to batch size
        avg_cls_token = avg_cls_token.expand(batch_size, -1, -1)  # Shape: [B, 1, D]
                
        # Create final feature vector - but only include patches inside the circle
        # patches_in_circle = valid_patches.sum().item()

        # Convert valid patches to flat index array for efficient selection
        flat_indices = torch.nonzero(valid_patches.view(-1)).squeeze()

        # Extract only active features from merged tensor
        B, H, W, D = merged.shape
        merged_view = merged.view(-1, D)  # Flatten spatial dimensions
        circle_features = merged_view[flat_indices]  # [num_valid_patches, D]

        # Reshape as if it were a normal feature tensor
        merged_flat = circle_features.view(B, -1, D)  # [B, num_valid_patches, D]

        # Store information about valid patch locations for later use
        self.valid_patch_mask = valid_patches  # Store for reference
            
        # Add CLS token
        merged_with_cls = torch.cat([avg_cls_token, merged_flat], dim=1)
        
        return merged_with_cls
    
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
    
    def forward(self, before_img, after_img):
        """Forward pass through the model with memory optimization for circular images"""
        # Clear previous mask information
        if hasattr(self, 'valid_patch_mask'):
            delattr(self, 'valid_patch_mask')
        
        # Set training mode
        is_training = self.training

        if self.feature_combination == 'diff_first':
            # Calculate difference at token level first, then pool
            before_features_full = self.extract_features_sliding_window(before_img, training=is_training)

            print("After feature extraction 1:")
            print_cuda_memory()

            after_features_full = self.extract_features_sliding_window(after_img, training=is_training)
            
            print("After feature extraction 2:")
            print_cuda_memory()

            # Calculate difference between token representations
            diff_features =  after_features_full - before_features_full
            
            print("After calculating difference:")
            print_cuda_memory()            

            # Free memory
            del before_features_full, after_features_full
            torch.cuda.empty_cache()

            # Apply pooling
            pooled_diff = self.pool_features(diff_features)

            print("After calculating pooled features:")
            print_cuda_memory()

            # Free memory
            del diff_features
            torch.cuda.empty_cache()

            logits = self.classifier(pooled_diff)

            # Free memory
            del pooled_diff
            torch.cuda.empty_cache()
            
            print("After final classification head:")
            print_cuda_memory()

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