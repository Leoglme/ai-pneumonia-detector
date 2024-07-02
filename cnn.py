import os
import pandas as pd
from utils.image_utils import ImageUtils
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np

# Disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import tensorflow as tf


class DataHandler:
    """
    This class handles the creation of data generators for training, validation, and testing datasets.
    """

    def __init__(self, data_dir, img_size=(256, 256), batch_sz=50):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_sz = batch_sz
        self.train_dir = os.path.join(data_dir, 'train')
        self.validation_dir = os.path.join(data_dir, 'val')
        self.train_generator = None
        self.validation_generator = None
        self.file_paths, self.image_stats = ImageUtils.filter_images(data_dir, img_size)
        self._create_generators()

    def _create_generators(self):
        """
        Create data generators for training and validation datasets.
        """
        train_df, val_df = self._create_dataframe(self.file_paths)

        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

        self.train_generator = datagen.flow_from_dataframe(
            train_df,
            x_col='filepath',
            y_col='class',
            target_size=self.img_size,
            batch_size=self.batch_sz,
            class_mode='binary'
        )

        self.validation_generator = datagen.flow_from_dataframe(
            val_df,
            x_col='filepath',
            y_col='class',
            target_size=self.img_size,
            batch_size=self.batch_sz,
            class_mode='binary'
        )

    @staticmethod
    def _create_dataframe(file_paths):
        """
        Create a dataframe from the list of file_paths.
        """
        data = {
            'filepath': file_paths,
            'class': ['PNEUMONIA' if 'PNEUMONIA' in fp else 'NORMAL' for fp in file_paths]
        }
        df = pd.DataFrame(data)
        train_df = df.sample(frac=0.8, random_state=42)
        val_df = df.drop(train_df.index)
        return train_df, val_df

    def get_dataset(self, generator):
        return tf.data.Dataset.from_generator(
            lambda: generator,
            output_signature=(
                tf.TensorSpec(shape=(None, *self.img_size, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32)
            )
        ).repeat()


class PneumoniaDetector:
    """
    This class defines the CNN model for pneumonia detection and provides methods for training and evaluation.
    """

    def __init__(self, input_shape):
        self.model = self._build_model(input_shape)

    @staticmethod
    def _build_model(input_shape):
        """
        Build the CNN model.
        """
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self, train_ds, val_ds, epochs=10, train_steps=100, val_steps=50):
        """
        Train the model using the training and validation data generators.
        """
        history = self.model.fit(
            train_ds,
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_data=val_ds,
            validation_steps=val_steps
        )
        self.model.summary()
        return history

    def evaluate(self, test_ds, test_steps):
        """
        Evaluate the model using the test data generator.
        """
        loss, accuracy = self.model.evaluate(test_ds, steps=test_steps)
        y_true = []
        y_pred = []
        y_proba = []

        for x, y in test_ds.take(test_steps):
            y_true.extend(y.numpy())
            preds = self.model.predict(x)
            y_pred.extend(np.where(preds >= 0.5, 1, 0))
            y_proba.extend(preds)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred).flatten()
        y_proba = np.array(y_proba).flatten()

        cm = confusion_matrix(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_proba)
        fpr, tpr, _ = roc_curve(y_true, y_proba)

        return loss, accuracy, cm, roc_auc, fpr, tpr

    def plot_roc_curve(self, fpr, tpr, roc_auc):
        """
        Plot the ROC curve.
        """
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

    def plot_confusion_matrix(self, cm):
        """
        Plot the confusion matrix.
        """
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NORMAL', 'PNEUMONIA'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.show()

    def load_model(self, model_path):
        """
        Load the model from the specified path.
        """
        self.model = tf.keras.models.load_model(model_path)


if __name__ == "__main__":
    data_directory = 'datasets'
    image_size = (256, 256)
    batch_size = 32
    num_epochs = 10  # 10
    steps_per_epoch = 50  # 50
    validation_steps = 50  # 50
    test_steps = 50  # 50

    # Initialize data handlers
    data_handler = DataHandler(data_directory, image_size, batch_size)

    # Print image statistics
    print(f"Total images: {data_handler.image_stats['total_images']}")
    print(f"Filtered images: {data_handler.image_stats['filtered_images']}")
    print(f"Min image size: {data_handler.image_stats['min_size']}")
    print(f"Max image size: {data_handler.image_stats['max_size']}")
    print(f"Average image width: {data_handler.image_stats['avg_width']}")
    print(f"Average image height: {data_handler.image_stats['avg_height']}")

    # Initialize the model
    detector = PneumoniaDetector(input_shape=(image_size[0], image_size[1], 3))

    # Get datasets
    train_ds = data_handler.get_dataset(data_handler.train_generator)
    val_ds = data_handler.get_dataset(data_handler.validation_generator)

    # Train the model
    history = detector.train(train_ds, val_ds, num_epochs, steps_per_epoch, validation_steps)

    # Evaluate the model
    test_loss, test_acc, cm, roc_auc, fpr, tpr = detector.evaluate(val_ds, test_steps)
    print(f'Test accuracy: {test_acc * 100:.2f}%')
    print(f'ROC AUC Score: {roc_auc}')
    print("Confusion Matrix:")
    print(cm)

    # Save the results to a file
    results = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist()
    }
    with open('cnn_results.json', 'w') as f:
        json.dump(results, f)

    # Plot ROC curve
    detector.plot_roc_curve(fpr, tpr, roc_auc)

    # Plot Confusion Matrix
    detector.plot_confusion_matrix(cm)