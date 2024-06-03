import os

import tensorflow as tf
from tensorflow.keras import layers

from dev.stats import affichage_courbe, basic_stat

base_chemin = '../../../../datasets/chest_Xray/'

# Définir la taille des lots
batch_size = 32  # Réduire la taille des lots


def load_data(dir):  # Modifier la taille des lots
    # Chargement des données
    return tf.keras.utils.image_dataset_from_directory(
        os.path.join(base_chemin, dir),
        labels='inferred',
        label_mode='int',
        image_size=(128, 128),
        batch_size=batch_size,
        shuffle=True,
    )
    basic_stat(base_chemin, dir)


# Création du modèle
model = tf.keras.models.Sequential([
    layers.Rescaling(1. / 255, input_shape=(128, 128, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compilation du modèle
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

# Entraînement du modèle
train_ds = load_data("train")
val_ds = load_data("val")
history = model.fit(
    train_ds,
    steps_per_epoch=100,
    epochs=20,
    validation_data=val_ds
)
# Test validation
test_ds = load_data("test")
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Loss sur l'ensemble de test: {test_loss}")
print(f"Précision sur l'ensemble de test: {test_accuracy}")

affichage_courbe(history)
