# Change Detection Project

This repository contains code for a satellite image change detection project that uses deep learning techniques to identify whether a conflict event has taken place between images taken at different times.

## Project Structure

- **Main Scripts**:
  - `train.py` - Main training script for change detection models (including fine-tuning)
  - `train_classifier_on_diff.py` - Trains the classifier head on already extracted difference features
  - `gridsearch.py` - Hyperparameter optimization using grid search
  - `concatenate_diff_features.py` - Combines different feature sets
  - `feature_extractor_2.py` - Extracts features from images

- **Data Handling**:
  - `cd_dataset.py` - Dataset class for change detection data
  - `change_detection_model.py` - Model architectures for change detection

- **Analysis Notebooks**:
  - `batch_size_calc.ipynb` - Calculates feasible batch sizes given resource constraints
  - `examine_models.ipynb` - Analysis and visualization of model performance

- **Directories**:
  - `acquire_data` - Scripts for data acquisition
  - `demo` - Demo applications 
  - `process_images` - Image data processing utilities
  - `util` - General utility functions
  - `vitae_models` - ViTAE model implementations

## Data
Conflict event data comes from the Armed Conflict Location & Event Data (ACLED) project. The data covers the Russo-Ukrainian war (2014-2024).

Satellite data comes from Planet Labs PBC.

## Results

F1 = 0.88
Accuracy = 0.85

## Requirements

- Python 3.9+
- see requirements.txt

## Credit

- ViTAE models come from https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA
