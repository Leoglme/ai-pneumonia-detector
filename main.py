import os

# Désactiver les optimisations de oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Masquer les warnings de TensorFlow
import tensorflow as tf

# Définir les chemins vers les répertoires de données
base_dir = 'datasets/chest_Xray'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Création des générateurs de données pour normaliser les images (les valeurs des pixels seront entre 0 et 1)
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

# Générateur de données pour l'ensemble d'entraînement
train_generator = train_datagen.flow_from_directory(
    train_dir,  # Répertoire de données d'entraînement
    target_size=(128, 128),  # Redimensionner les images à 128x128 pixels
    batch_size=20,  # Nombre d'images à traiter par lot
    class_mode='binary')  # Les étiquettes sont binaires (NORMAL ou PNEUMONIA)

# Générateur de données pour l'ensemble de validation
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,  # Répertoire de données de validation
    target_size=(128, 128),  # Redimensionner les images à 128x128 pixels
    batch_size=20,  # Nombre d'images à traiter par lot
    class_mode='binary')  # Les étiquettes sont binaires (NORMAL ou PNEUMONIA)

# Générateur de données pour l'ensemble de test
test_generator = test_datagen.flow_from_directory(
    test_dir,  # Répertoire de données de test
    target_size=(128, 128),  # Redimensionner les images à 128x128 pixels
    batch_size=20,  # Nombre d'images à traiter par lot
    class_mode='binary')  # Les étiquettes sont binaires (NORMAL ou PNEUMONIA)

# Définition du modèle CNN (réseau de neurones convolutif)
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(128, 128, 3)),  # Couche d'entrée
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    # Couche de convolution avec 32 filtres, taille de filtre 3x3, et fonction d'activation ReLU
    tf.keras.layers.MaxPooling2D((2, 2)),  # Couche de pooling avec une fenêtre de 2x2
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),  # Deuxième couche de convolution avec 64 filtres
    tf.keras.layers.MaxPooling2D((2, 2)),  # Deuxième couche de pooling
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),  # Troisième couche de convolution avec 128 filtres
    tf.keras.layers.MaxPooling2D((2, 2)),  # Troisième couche de pooling
    tf.keras.layers.Flatten(),  # Aplatir la sortie 3D en une seule dimension pour la couche dense
    tf.keras.layers.Dense(512, activation='relu'),  # Couche dense avec 512 neurones et activation ReLU
    tf.keras.layers.Dense(1, activation='sigmoid')
    # Couche de sortie avec un neurone et activation sigmoïde pour la classification binaire
])

# Compilation du modèle avec l'optimiseur Adam et la fonction de perte binaire cross-entropie
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Entraînement du modèle
history = model.fit(
    train_generator,  # Générateur de données d'entraînement
    steps_per_epoch=10,  # Nombre d'étapes par époque
    epochs=5,  # Nombre d'époques (périodes d'entraînement)
    validation_data=validation_generator,  # Générateur de données de validation
    workers=4,  # Nombre de processus de travail (workers) à utiliser
    validation_steps=50  # Nombre d'étapes de validation par époque
)

# Évaluation du modèle sur l'ensemble de test
test_loss, test_acc = model.evaluate(test_generator, steps=50)  # Évaluer le modèle sur 50 étapes de test
print(f'Test accuracy: {test_acc * 100:.2f}%')  # Afficher la précision du modèle sur l'ensemble de test
