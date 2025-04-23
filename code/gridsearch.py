#!/usr/bin/env python3

import subprocess
import itertools
import os
import pandas as pd
import time
from datetime import datetime
import argparse
from train_classifier_on_diff import finetune_change_detection_model

def run_grid_search(
    learning_rates=[0.05, 0.01, 0.001, 0.0001, 0.00001],
    l1_lambdas=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
    window_sizes=[512],
    overlaps=[32],
    dropout1=[0.1, 0.2, 0.3, 0.4, 0.5],
    dropout2=[0.1, 0.2, 0.3, 0.4, 0.5],
    feature_poolings=["max"],
    feature_combinations=["diff_first"],
    classifier_types=["linear", "mlp"],
    batch_size=[50, 100, 200, 500],
    epochs_classifier=[40],
    epochs_finetuning=[3],
    gradient_accumulation_steps=1,
    # before_path="../data/images_ukraine_extracted_before/",
    # after_path="../data/images_ukraine_extracted_after/",
    annotations_path="../data/annotations_ukraine_n_1600.csv",
    checkpoint_path="../data/model_weights/vit-b-checkpoint-1599.pth",
    output_dir="../data/model_results/"
):
    """
    Run grid search over learning rates and L1 lambdas.
    
    Args:
        learning_rates: List of learning rates to try
        l1_lambdas: List of L1 regularization strengths to try
        window_sizes: List of window sizes to try
        overlaps: List of window overlaps to try
        feature_poolings: List of feature pooling methods to try
        feature_combinations: List of feature combination methods to try
        classifier_types: List of classifier types to try
        batch_size: Batch size for training
        epochs_classifier: Number of epochs for classifier training
        epochs_finetuning: Number of epochs for finetuning
        gradient_accumulation_steps: Gradient accumulation steps
        before_path: Path to 'before' images
        after_path: Path to 'after' images
        annotations_path: Path to annotations CSV
        checkpoint_path: Path to pretrained MAE checkpoint
        output_dir: Directory to save finetuned models
    """
    # Generate parameter combinations
    param_combinations = list(itertools.product(
        learning_rates,
        l1_lambdas,
        window_sizes,
        overlaps,
        dropout1,
        dropout2,
        feature_poolings,
        feature_combinations,
        classifier_types,
        batch_size
    ))
    results = []

    
    
    
    print(f"Starting grid search with {len(param_combinations)} parameter combinations")
    
    for i, (lr, l1, win_size, overlap, do1, do2, pooling, combination, classifier, batch_size) in enumerate(param_combinations):
        # Create a timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\n{'='*80}")
        print(f"Running combination {i+1}/{len(param_combinations)}:")
        print(f"  Learning rate: {lr}")
        print(f"  L1 lambda: {l1}")
        print(f"  Window size: {win_size}")
        print(f"  Dropout1: {do1}")
        print(f"  Dropout2: {do2}")
        print(f"  Overlap: {overlap}")
        print(f"  Feature pooling: {pooling}")
        print(f"  Feature combination: {combination}")
        print(f"  Classifier type: {classifier}")
        print(f"  Batch size: {batch_size}")
        print(f"{'='*80}\n")

        # check if output directory exists, if not create it
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Define model output path for this run
        model_path = os.path.join(output_dir, f"model_{timestamp}_lr{lr}_l1{l1}.pth")
        
        # Run the training script
        try:
            _, val_loss, accuracy, f1 = finetune_change_detection_model(
                # before_path=before_path,
                # after_path=after_path,
                annotations_path=annotations_path,
                # checkpoint_path=checkpoint_path,
                # model_path=model_path,
                learning_rate=lr,
                l1_lambda=l1,
                dropout1=do1,
                dropout2=do2,
                # window_size=win_size,
                # overlap=overlap,
                # feature_pooling=pooling,
                # feature_combination=combination,
                classifier_type=classifier,
                batch_size=batch_size,
                epochs_classifier=epochs_classifier,
                # epochs_finetuning=epochs_finetuning,
                gradient_accumulation_steps=gradient_accumulation_steps
            )
            
            # Append results to the list
            results.append({
                "learning_rate": lr,
                "l1_lambda": l1,
                "dropout1": do1,
                "dropout2": do2,
                "window_size": win_size,
                "overlap": overlap,
                "feature_pooling": pooling,
                "feature_combination": combination,
                "classifier_type": classifier,
                "batch_size": batch_size,
                "val_loss": val_loss,
                "accuracy": accuracy,
                "f1_score": f1
            })

            
        except Exception as e:
            print(f"Error running training for combination: {e}")
            continue

    # Save all results to a single CSV file
    results_df = pd.DataFrame(results)
    combined_results_csv_path = os.path.join(output_dir, f"combined_results_{timestamp}.csv")
    results_df.to_csv(combined_results_csv_path, index=False)
    print(f"Combined results saved to {combined_results_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run grid search for change detection model")
    parser.add_argument("--annotations_path", type=str, default="../data/feature_annotations.csv", 
                       help="Path to annotations CSV")
    args = parser.parse_args()
    
    # Define parameter grid
    learning_rates=[0.001]
    window_sizes=[512]
    overlaps=[32]
    batch_size=[100]
    epochs_classifier=25
    l1_lambdas=[0.01]
    feature_poolings = ["max"]
    feature_combinations = ["diff_first"]
    classifier_types = ["mlp"]
    dropout1 = [0.1, 0.2, 0.3, 0.4, 0.5]
    dropout2 = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Run grid search with sensible defaults
    run_grid_search(
        learning_rates=learning_rates,
        l1_lambdas=l1_lambdas,
        feature_poolings=feature_poolings,
        feature_combinations=feature_combinations,
        classifier_types=classifier_types,
        annotations_path=args.annotations_path,
        window_sizes=window_sizes,
        overlaps=overlaps,
        dropout1=dropout1,
        dropout2=dropout2,
        batch_size=batch_size,
        epochs_classifier=epochs_classifier
    )