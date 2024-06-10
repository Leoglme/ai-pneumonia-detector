import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, log_loss
import tensorflow as tf


# Charger les jeux de données
train_dataset = tf.keras.utils.image_dataset_from_directory(
    'chest_Xray/train',
    image_size=(150, 150),
    batch_size=32,
    label_mode='binary'
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    'chest_Xray/val',
    image_size=(150, 150),
    batch_size=32,
    label_mode='binary'
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    'chest_Xray/test',
    image_size=(150, 150),
    batch_size=32,
    label_mode='binary'
)

# Fonction pour convertir le dataset en numpy arrays
def dataset_to_numpy(dataset):
    images = []
    labels = []
    for image, label in dataset.unbatch():
        images.append(image.numpy())
        labels.append(label.numpy())
    return np.array(images), np.array(labels)

train_images, train_labels = dataset_to_numpy(train_dataset)

# Normaliser les images
train_images = train_images / 255.0

# Remodeler en tableau de 1D pour KNN
train_images_flat = train_images.reshape((train_images.shape[0], -1))

# Validation croisée
kf = KFold(n_splits=5)
fold_results = []

for train_index, val_index in kf.split(train_images_flat):
    x_train, x_val = train_images_flat[train_index], train_images_flat[val_index]
    y_train, y_val = train_labels[train_index], train_labels[val_index]
    
    # Créer et entraîner le modèle KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)
    
    #### Évaluer le modèle
    # Prédir les étiquettes pour les données de validation
    y_val_pred = knn.predict(x_val)
    # Probabilités prédictives des classes pour les données de validation
    y_val_proba = knn.predict_proba(x_val)[:, 1]
    # Comparer les étiquettes prédites y_val_pred avec les étiquettes réelles y_val
    val_acc = accuracy_score(y_val, y_val_pred)
    # Calculer la log-perte (ou log-loss) pour les prédictions du modèle sur les données de validation.
    val_loss = log_loss(y_val, y_val_proba)
    
    # Stocker les résultats de chaque itération de validation croisée dans une liste
    fold_results.append((val_loss, val_acc))
    print(f'Validation accuracy: {val_acc}, Validation loss: {val_loss}')

# Moyenne des résultats de la validation croisée
avg_val_loss = np.mean([result[0] for result in fold_results])
avg_val_acc = np.mean([result[1] for result in fold_results])
print(f'Average validation accuracy (KFold): {avg_val_acc}, Average validation loss (KFold): {avg_val_loss}')

# Comparaison avec une simple séparation train-test
x_train, x_val, y_train, y_val = train_test_split(train_images_flat, train_labels, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

y_val_pred = knn.predict(x_val)
y_val_proba = knn.predict_proba(x_val)[:, 1]
val_acc = accuracy_score(y_val, y_val_pred)
val_loss = log_loss(y_val, y_val_proba)
print(f'Simple train-test split validation accuracy: {val_acc}, validation loss: {val_loss}')

# Évaluation finale sur le jeu de test
test_images, test_labels = dataset_to_numpy(test_dataset)
test_images_flat = test_images.reshape((test_images.shape[0], -1))
y_test_pred = knn.predict(test_images_flat)
y_test_proba = knn.predict_proba(test_images_flat)[:, 1]
test_acc = accuracy_score(test_labels, y_test_pred)
test_loss = log_loss(test_labels, y_test_proba)
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')
