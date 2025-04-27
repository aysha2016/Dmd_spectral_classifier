
from load_data import load_data
from dmd_photonic_transform import generate_spectral_data
from unet_model import unet_for_classification
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = load_data()
x_train_spec = generate_spectral_data(x_train, max_samples=2000)
y_train_spec = y_train[:2000]
x_test_spec = generate_spectral_data(x_test, max_samples=500)
y_test_spec = y_test[:500]

model = unet_for_classification(input_shape=(3, 64, 1), num_classes=10)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x_train_spec, y_train_spec, epochs=10, batch_size=32, validation_data=(x_test_spec, y_test_spec))
