# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a time series forecasting project using LSTM recurrent neural networks to predict temperature from the Jena Climate dataset (2009-2016). The project demonstrates temperature prediction 24 hours ahead using meteorological features.

## Project Structure

- `Untitled-1.py`: Main training script with LSTM model implementation
- `jena_climate_2009_2016.csv`: Climate dataset with 14 meteorological features (pressure, temperature, humidity, wind, etc.) recorded every 10 minutes
- `jena_dense.keras`: Saved model checkpoint (best performing model)

## Key Architecture Details

**Data Processing Pipeline:**
- Train/Val/Test split: 50%/25%/25% of total data
- Normalization: Z-score normalization using only training data statistics
- Sampling rate: 6 (using every 6th observation = 1-hour intervals)
- Sequence length: 120 timesteps (120 hours of history)
- Prediction target: Temperature 24 hours ahead (delay parameter)
- Batch size: 256

**Model Architecture:**
- Input: Sequences of shape `(sequence_length, num_features)` where num_features = 14
- LSTM layer: 32 units with 25% recurrent dropout
- Dropout layer: 50%
- Dense output: Single value (temperature prediction)
- Optimizer: RMSprop
- Loss: MSE (Mean Squared Error)
- Metric: MAE (Mean Absolute Error)

**Baseline Method:**
The code includes a naive forecasting baseline (lines 68-78) that uses the last observed temperature value as the prediction, denormalized for comparison.

## Running the Code

**Train the model:**
```bash
python Untitled-1.py
```

This will:
1. Load and preprocess the Jena climate data
2. Create train/val/test datasets
3. Evaluate naive baseline method
4. Train LSTM model for 10 epochs
5. Save best model to `jena_dense.keras`
6. Display training/validation MAE plots

**Dependencies:**
- tensorflow/keras
- numpy
- matplotlib

## Important Implementation Notes

- The manual RNN implementation (lines 81-95) is for educational purposes showing how recurrent state updates work
- Temperature is in index position 1 of the features array (column 2 in CSV)
- Data normalization statistics (mean/std) are computed only on training set to prevent data leakage
- ModelCheckpoint callback saves only the best model based on validation loss
