# Building Damage Segmentation

A deep learning project that automatically identifies and classifies building damage from satellite images.

## Project Goal

The model analyzes aerial images and detects different levels of building damage, helping with disaster assessment and response.

## How It Works

- **Input**: Satellite images of buildings
- **Process**: U-Net neural network analyzes the images
- **Output**: Damage maps showing 5 damage levels:
  - No damage
  - Minor damage
  - Major damage
  - Destroyed
  - Total destruction

## Training

The model was trained on 500 images for 50 epochs, achieving 42% accuracy in damage detection.

## Usage

```bash
python Deep_Learning.py



View training progress:
```bash
tensorboard --logdir=logs_[timestamp]/

## Results

https://sample_comparison.png
https://training_curves.png