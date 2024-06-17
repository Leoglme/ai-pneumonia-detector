import pandas as pd
from utils.image_utils import ImageUtils
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
from PIL import Image


class DataHandler:
    """
    This class handles the creation of datasets for training and validation.
    """

    def __init__(self, data_dir, img_size=(128, 128)):
        self.data_dir = data_dir
        self.img_size = img_size
        self.file_paths, self.image_stats = ImageUtils.filter_images(data_dir, img_size)
        self.train_df, self.val_df = self._create_dataframe(self.file_paths)
        self.X_train, self.X_val, self.y_train, self.y_val = self._create_datasets()

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

    def _load_image(self, filepath):
        """
        Load and preprocess an image.
        """
        with Image.open(filepath) as img:
            img = img.resize(self.img_size)
            img_array = np.array(img)
            # Ensure the image is 3 channels
            if img_array.ndim == 2:
                img_array = np.stack((img_array,)*3, axis=-1)
            elif img_array.shape[2] == 1:
                img_array = np.concatenate((img_array, img_array, img_array), axis=-1)
            return img_array.flatten() / 255.0

    def _create_datasets(self):
        """
        Create datasets for training and validation.
        """
        X_train = np.array([self._load_image(fp) for fp in self.train_df['filepath']])
        y_train = np.array([1 if label == 'PNEUMONIA' else 0 for label in self.train_df['class']])

        X_val = np.array([self._load_image(fp) for fp in self.val_df['filepath']])
        y_val = np.array([1 if label == 'PNEUMONIA' else 0 for label in self.val_df['class']])

        return X_train, X_val, y_train, y_val


class PneumoniaDetectorKNN:
    """
    This class defines the KNN model for pneumonia detection and provides methods for training and evaluation.
    """

    def __init__(self, n_neighbors=5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        """
        Train the KNN model using the training dataset.
        """
        X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train)

    def evaluate(self, X_val, y_val):
        """
        Evaluate the model using the validation dataset.
        """
        X_val = self.scaler.transform(X_val)
        y_pred = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred, target_names=['NORMAL', 'PNEUMONIA'])
        cm = confusion_matrix(y_val, y_pred)
        return accuracy, report, cm


if __name__ == "__main__":
    data_directory = 'datasets'
    image_size = (64, 64)  # Smaller images for KNN
    num_neighbors = 5

    # Initialize data handlers
    data_handler = DataHandler(data_directory, image_size)

    # Print image statistics
    print(f"Total images: {data_handler.image_stats['total_images']}")
    print(f"Filtered images: {data_handler.image_stats['filtered_images']}")
    print(f"Min image size: {data_handler.image_stats['min_size']}")
    print(f"Max image size: {data_handler.image_stats['max_size']}")
    print(f"Average image width: {data_handler.image_stats['avg_width']}")
    print(f"Average image height: {data_handler.image_stats['avg_height']}")

    # Print dataset shapes
    print(f"Training set shape: {data_handler.X_train.shape}, Validation set shape: {data_handler.X_val.shape}")

    # Initialize the model
    detector = PneumoniaDetectorKNN(n_neighbors=num_neighbors)

    # Train the model
    detector.train(data_handler.X_train, data_handler.y_train)

    # Evaluate the model
    accuracy, report, cm = detector.evaluate(data_handler.X_val, data_handler.y_val)
    print(f'Validation accuracy: {accuracy * 100:.2f}%')
    print(report)
    print("Confusion Matrix:")
    print(cm)
