
# 📦 DMD + Photonic Spectral Classification with U-Net

This project simulates a **photonic spectral transformation pipeline** for classifying CIFAR-10 images using a **U-Net-inspired model** adapted for classification.

## 📁 Project Structure

```
dmd_spectral_classifier/
├── load_data.py                  # Loads and preprocesses CIFAR-10 dataset
├── dmd_photonic_transform.py     # Applies DMD + photonic spectral transformation
├── unet_model.py                 # Defines U-Net encoder + classification model
├── train.py                      # Trains and evaluates the model
└── README.md                     # This file
```

## 🧪 Requirements

Install dependencies via:

```bash
pip install tensorflow numpy
```

## 🚀 How to Run

1. **Train the Model:**

```bash
python train.py
```

2. **Adjust Parameters:**

Inside `train.py`, you can change:
- `max_samples`: Number of CIFAR-10 samples to simulate
- `epochs`, `batch_size`: Training config

## 🧠 Model Overview

- **Input**: Simulated spectral output of RGB images (shape: `(3, 64, 1)`)
- **Encoder**: U-Net style convolution blocks
- **Classifier**: Dense layers replacing the decoder
- **Output**: 10-way softmax classification

## 🔬 Notes

- The `photonic_layer_rgb` uses a random projection to simulate optical hardware.
- You can extend this with t-SNE, PCA, or confusion matrix visualization.
- Optionally, replace U-Net with other architectures (e.g., ResNet, VAE).
