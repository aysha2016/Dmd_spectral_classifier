
import numpy as np

def dmd_binary_pattern(image):
    return (image > 0.5).astype(np.float32)

def photonic_layer_rgb(img, n_outputs=64):
    flattened = [img[:, :, i].flatten() for i in range(3)]
    spectral_outputs = []
    for flat in flattened:
        proj_matrix = np.random.randn(n_outputs, flat.size)
        output = proj_matrix @ flat
        spectral_outputs.append(output)
    return np.stack(spectral_outputs, axis=0).reshape((3, n_outputs, 1))

def generate_spectral_data(data, max_samples=1000):
    spec_data = []
    for i in range(max_samples):
        dmd = dmd_binary_pattern(data[i])
        spectral = photonic_layer_rgb(dmd)
        spec_data.append(spectral)
    return np.array(spec_data)
