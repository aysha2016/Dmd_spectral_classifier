# Photonic Spectral Transformation Pipeline for CIFAR-10 Classification

# 📦 DMD + Photonic Spectral Classification with U-Net

This project simulates a photonic spectral transformation (using FFT) on CIFAR-10 images and then trains a modified U‑Net‑inspired model (adapted for classification) on the transformed data. The pipeline consists of the following steps:

- **Photonic Spectral Transformation (using FFT):**  
  The photonic transformation (simulated via FFT) is applied to each channel of the CIFAR‑10 images (see `dmd_photonic_transform.py`).  
- **Modified U‑Net‑Inspired Model:**  
  A U‑Net‑inspired model (adapted for classification) is defined (see `unet_model.py`).  
- **Data Loading and Preprocessing:**  
  CIFAR‑10 data is loaded, scaled (normalized) and one‑hot encoded (see `load_data.py`).  
- **Training and Evaluation:**  
  The photonic transformation is applied (using `generate_spectral_data`), the model is instantiated, compiled, and then trained (and evaluated) (see `train.py`).

## Files

- **dmd_photonic_transform.py**  
  Simulates a photonic spectral transformation (using FFT) on each channel of the image.  
  - `dmd_binary_pattern(image)`: Converts an image into a binary pattern (using a threshold).  
  - `photonic_layer_rgb_fft(img, n_outputs=64)`: Applies FFT (using `np.fft.fft2`) to each channel, computes the magnitude, and (optionally) selects the largest components.  
  - `generate_spectral_data(data, max_samples=1000)`: Applies the photonic transformation (via `photonic_layer_rgb_fft`) on a batch of images (up to a maximum number of samples).

- **unet_model.py**  
  Defines a modified U‑Net‑inspired model (adapted for classification).  
  - `unet_for_classification(input_shape, num_classes)`: Builds a model (using TensorFlow/Keras) that consists of convolutional layers (with ReLU activation and padding), max pooling, global average pooling, and dense layers (with softmax output for classification).

- **load_data.py**  
  Loads and preprocesses CIFAR‑10 data.  
  - `load_data()`: Calls `cifar10.load_data()`, scales the images (dividing by 255.0), and one‑hot encodes the labels (using `tf.keras.utils.to_categorical`).

- **train.py**  
  Ties the pipeline together.  
  - Loads (and preprocesses) CIFAR‑10 data (using `load_data`).  
  - Applies the photonic spectral transformation (using `generate_spectral_data`) on training and test data (with a maximum of 2000 training samples and 500 test samples).  
  - Instantiates the modified U‑Net‑inspired model (using `unet_for_classification`), compiles it (using Adam optimizer and categorical cross‑entropy loss), and then trains (and evaluates) the model (using `model.fit`).

## Running the Pipeline

1. **Install Dependencies:**  
   Ensure that you have installed the following Python packages (for example, via pip):  
   - numpy  
   - tensorflow (or tensorflow‑gpu)  
   (You may also use a virtual environment.)

2. **Run the Training Script:**  
   Execute the training script (e.g. via the command line) as follows:  
   ```bash
   python train.py
   ```  
   This will load CIFAR‑10, apply the photonic transformation (using FFT), instantiate the U‑Net‑inspired model, and then train (and evaluate) the model.

.
