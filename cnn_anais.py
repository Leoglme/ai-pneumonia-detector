import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Load datasets with different batch sizes
def load_data(directory, batch_size=32, label_mode='binary', image_size=(150, 150)):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        batch_size=batch_size,
        label_mode=label_mode,
        image_size=image_size
    )

train_dataset = load_data('datasets/train', batch_size=32)
validation_dataset = load_data('datasets/val', batch_size=16)  # Adjust batch size for small validation set
test_dataset = load_data('datasets/test', batch_size=16)  # Adjust batch size for small test set

# Preprocessing function remains the same
def preprocess_image(image, label):
    image = image / 255.0  # Normalize to [0,1] range
    return image, label

train_dataset = train_dataset.map(preprocess_image)
validation_dataset = validation_dataset.map(preprocess_image)
test_dataset = test_dataset.map(preprocess_image)

# Define the model
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(150, 150, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Convert dataset to numpy arrays remains the same
def dataset_to_numpy(dataset):
    images = []
    labels = []
    for image, label in dataset.unbatch().take(1000):
        images.append(image.numpy())
        labels.append(label.numpy())
    return np.array(images), np.array(labels)

train_data, train_labels = dataset_to_numpy(train_dataset)
val_data, val_labels = dataset_to_numpy(validation_dataset)
test_data, test_labels = dataset_to_numpy(test_dataset)

# Utilisation de la validation croisée
kf = KFold(n_splits=5, shuffle=True, random_state=42)
val_acc = []
batch_size = 32  # Ajustez la taille du lot si nécessaire

for train_index, val_index in kf.split(train_data):
    x_train, x_val = train_data[train_index], train_data[val_index]
    y_train, y_val = train_labels[train_index], train_labels[val_index]
    model = create_model()
    model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val), batch_size=batch_size)
    val_acc.append(model.evaluate(x_val, y_val)[1])

print(f'Validation Accuracy: {np.mean(val_acc)}')

# Entraînement final
model = create_model()
model.fit(train_data, train_labels, epochs=10, validation_data=(val_data, val_labels), batch_size=batch_size)

# Évaluation finale
test_loss, test_acc = model.evaluate(test_data, test_labels)
print(f'Test Accuracy: {test_acc}')

# Comparaison avec une simple division train-test
x_train, x_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)
model = create_model()
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), batch_size=batch_size)
test_loss, test_acc = model.evaluate(test_data, test_labels)
print(f'Test Accuracy with simple train-test split: {test_acc}')