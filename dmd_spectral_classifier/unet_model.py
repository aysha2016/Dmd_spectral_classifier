# Complete code for modified U-Net-inspired model (adapted for classification) screen (unet_model.py)

from tensorflow.keras import layers, models

def unet_for_classification(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(c3)
    flat = layers.GlobalAveragePooling2D()(c3)

    dense = layers.Dense(64, activation='relu')(flat)
    outputs = layers.Dense(num_classes, activation='softmax')(dense)

    return models.Model(inputs, outputs)
