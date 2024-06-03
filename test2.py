import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
from sklearn.model_selection import train_test_split


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

# Fonction pour créer le modèle de réseau de neurones convolutifs (CNN)
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

# Validation croisée
kf = KFold(n_splits=5)
fold_results = []


for train_index, val_index in kf.split(train_images):
    x_train, x_val = train_images[train_index], train_images[val_index]
    y_train, y_val = train_labels[train_index], train_labels[val_index]
    
    model = create_model()
    
    history = model.fit(
        x_train, y_train,
        epochs=10,
        validation_data=(x_val, y_val)
    )
    
    # Évaluer le modèle
    val_loss, val_acc = model.evaluate(x_val, y_val)
    fold_results.append((val_loss, val_acc))
    print(f'Validation accuracy: {val_acc}, Validation loss: {val_loss}')


# Moyenne des résultats de la validation croisée
avg_val_loss = np.mean([result[0] for result in fold_results])
avg_val_acc = np.mean([result[1] for result in fold_results])
print(f'Average validation accuracy (KFold): {avg_val_acc}, Average validation loss (KFold): {avg_val_loss}')

# Comparaison avec une simple séparation train-test
x_train, x_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

model = create_model()

history = model.fit(
    x_train, y_train,
    epochs=10,
    validation_data=(x_val, y_val)
)

val_loss, val_acc = model.evaluate(x_val, y_val)
print(f'Simple train-test split validation accuracy: {val_acc}, validation loss: {val_loss}')

# Évaluation finale sur le jeu de test
test_images, test_labels = dataset_to_numpy(test_dataset)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')

