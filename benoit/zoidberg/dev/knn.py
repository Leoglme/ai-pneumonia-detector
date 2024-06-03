import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def load_data_knn(dir):
    dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(base_chemin, dir),
        labels='inferred',
        label_mode='int',
        image_size=(256, 256),
        batch_size=None,  # Lots inutiles pour KNN
        shuffle=True,
    )

    images = []
    labels = []

    for image, label in dataset:
        images.append(tf.reshape(image, [-1]).numpy())  # Aplatir les images
        labels.append(label.numpy())

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


# Exemple d'utilisation
base_chemin = '../../../../datasets/chest_Xray/'

# Chargement des données d'entraînement
X_train, y_train = load_data_knn("train")

# Chargement des données de test
X_test, y_test = load_data_knn("test")

# Initialisation et entraînement du modèle k-NN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Prédiction et évaluation
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
