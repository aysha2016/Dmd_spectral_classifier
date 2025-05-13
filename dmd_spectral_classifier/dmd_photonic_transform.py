# Complete code for photonic spectral transformation (using FFT) screen (dmd_photonic_transform.py)

import numpy as np

def dmd_binary_pattern(image):
    return (image > 0.5).astype(np.float32)

def photonic_layer_rgb_fft(img, n_outputs=64):
    spectral_outputs = []
    for i in range(3):
        # Apply FFT to each channel
        channel_fft = np.fft.fft2(img[:, :, i])
        # Take magnitude and flatten
        mag = np.abs(channel_fft).flatten()
        # Optionally, select n_outputs largest components (or use all if n_outputs is large)
        if n_outputs < mag.size:
            idx = np.argsort(mag)[-n_outputs:]
            output = mag[idx]
        else:
            output = mag
        spectral_outputs.append(output)
    return np.stack(spectral_outputs, axis=0).reshape((3, n_outputs, 1))

def generate_spectral_data(data, max_samples=1000):
    spec_data = []
    for i in range(max_samples):
        dmd = dmd_binary_pattern(data[i])
        spectral = photonic_layer_rgb_fft(dmd)
        spec_data.append(spectral)
    return np.array(spec_data)
