import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, log_loss
import tensorflow as tf


# Load datasets
def load_data (directory, image_size=(150, 150), batch_size=32, label_mode='binary'):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=image_size,
        batch_size=batch_size,
        label_mode=label_mode
    )

train_dataset = load_data('chest_Xray/train')    
validation_dataset = load_data('chest_Xray/val') 
test_dataset = load_data('chest_Xray/test')   

# Function to convert dataset into numpy arrays
def dataset_to_numpy(dataset):
    images = []
    labels = []
    for image, label in dataset.unbatch():
        images.append(image.numpy())
        labels.append(label.numpy())
    return np.array(images), np.array(labels)

train_images, train_labels = dataset_to_numpy(train_dataset)
val_images, val_labels = dataset_to_numpy(validation_dataset)
test_images, test_labels = dataset_to_numpy(test_dataset)

# Normalize images
train_images = train_images / 255.0
val_images = val_images / 255.0
test_images = test_images / 255.0

# Reshape into 1D array for KNN
train_images_flat = train_images.reshape((train_images.shape[0], -1))
val_images_flat = val_images.reshape((val_images.shape[0], -1))
test_images_flat = test_images.reshape((test_images.shape[0], -1))

### Train-Validation-Test

# Train the model with train dataset
model = KNeighborsClassifier(n_neighbors=5)
model.fit(train_images_flat, train_labels)

# Evaluate the model with validation dataset
val_predictions = model.predict(val_images_flat)
val_accuracy = accuracy_score(val_labels, val_predictions)
val_log_loss = log_loss(val_labels, val_predictions)
print(f'Validation Accuracy: {val_accuracy}')
print(f'Validation Log Loss: {val_log_loss}')

# Test the model with test dataset
test_predictions = model.predict(test_images_flat)
test_accuracy = accuracy_score(test_labels, test_predictions)
test_log_loss = log_loss(test_labels, test_predictions)
print(f'Test Accuracy: {test_accuracy}')
print(f'Test Log Loss: {test_log_loss}')


###  Cross Validation 
kf = KFold(n_splits=5)
fold_results = []

for train_index, val_index in kf.split(train_images_flat):
    x_train, x_val = train_images_flat[train_index], train_images_flat[val_index]
    y_train, y_val = train_labels[train_index], train_labels[val_index]
    
    # Create and train the KNN model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)
    
    #### Evaluate the modele
    
    # Predict lables for validation data
    y_val_pred = knn.predict(x_val)
    # Predictive probabilities of classes for validation data
    y_val_proba = knn.predict_proba(x_val)[:, 1]
    # Compare predicted labels y_val_pred with actual labels y_val
    val_acc = accuracy_score(y_val, y_val_pred)
    # Calculate log-loss (or log-loss) for model predictions on validation data.
    val_loss = log_loss(y_val, y_val_proba)
    
    # Store the results of each cross-validation iteration in a list
    fold_results.append((val_loss, val_acc))
    print(f'Validation accuracy: {val_acc}, Validation loss: {val_loss}')

# Average cross-validation results
avg_val_loss = np.mean([result[0] for result in fold_results])
avg_val_acc = np.mean([result[1] for result in fold_results])
print(f'Average validation accuracy (KFold): {avg_val_acc}, Average validation loss (KFold): {avg_val_loss}')

###  Compare with a simple train-test split
x_train, x_val, y_train, y_val = train_test_split(train_images_flat, train_labels, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

y_val_pred = knn.predict(x_val)
y_val_proba = knn.predict_proba(x_val)[:, 1]
val_acc = accuracy_score(y_val, y_val_pred)
val_loss = log_loss(y_val, y_val_proba)
print(f'Simple train-test split validation accuracy: {val_acc}, validation loss: {val_loss}')

# Final evaluation on the test game
val_images, val_labels = dataset_to_numpy(validation_dataset)
val_images_flat = val_images.reshape((val_images.shape[0], -1))
y_val_pred = knn.predict(val_images_flat)
y_val_proba = knn.predict_proba(val_images_flat)[:, 1]
val_acc = accuracy_score(val_labels, y_val_pred)
val_loss = log_loss(val_labels, y_val_proba)
print(f'Test accuracy: {val_acc}, Test loss: {val_loss}')