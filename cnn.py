import os
from PIL import Image
import pandas as pd

# Disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import tensorflow as tf


class DataHandler:
    """
    This class handles the creation of data generators for training, validation, and testing datasets.
    """

    def __init__(self, data_dir, img_size=(235, 235), batch_sz=50, min_img_size=(256, 256)):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_sz = batch_sz
        self.min_img_size = min_img_size
        self.train_dir = os.path.join(data_dir, 'train')
        self.validation_dir = os.path.join(data_dir, 'val')
        self.train_generator = None
        self.validation_generator = None
        self.image_stats = {
            'total_images': 0,
            'min_size': (float('inf'), float('inf')),
            'max_size': (0, 0),
            'avg_width': 0,
            'avg_height': 0,
            'filtered_images': 0
        }
        self.filepaths = []
        self._filter_images()
        self._create_generators()

    def _filter_images(self):
        """
        Filter out images that are too small and gather statistics on the image sizes.
        """
        total_width, total_height = 0, 0

        for dirpath, _, filenames in os.walk(self.data_dir):
            for filename in filenames:
                if filename.lower().endswith(('png', 'jpg', 'jpeg')):
                    filepath = os.path.join(dirpath, filename)
                    with Image.open(filepath) as img:
                        width, height = img.size
                        self.image_stats['total_images'] += 1
                        total_width += width
                        total_height += height
                        if width < self.min_img_size[0] or height < self.min_img_size[1]:
                            self.image_stats['filtered_images'] += 1
                        else:
                            self.filepaths.append(filepath)
                            self.image_stats['min_size'] = (
                                min(self.image_stats['min_size'][0], width),
                                min(self.image_stats['min_size'][1], height)
                            )
                            self.image_stats['max_size'] = (
                                max(self.image_stats['max_size'][0], width),
                                max(self.image_stats['max_size'][1], height)
                            )
                            self.image_stats['avg_width'] += width
                            self.image_stats['avg_height'] += height

        remaining_images = self.image_stats['total_images'] - self.image_stats['filtered_images']
        if remaining_images > 0:
            self.image_stats['avg_width'] /= remaining_images
            self.image_stats['avg_height'] /= remaining_images

        self.image_stats['avg_width'] = round(self.image_stats['avg_width'])
        self.image_stats['avg_height'] = round(self.image_stats['avg_height'])

        avg_width = self.image_stats['avg_width']
        avg_height = self.image_stats['avg_height']

        # Second pass to filter images smaller than the 20% of the average size
        filtered_filepaths = []
        for filepath in self.filepaths:
            with Image.open(filepath) as img:
                width, height = img.size
                if width < avg_width * 0.8 or height < avg_height * 0.8:
                    self.image_stats['filtered_images'] += 1
                else:
                    filtered_filepaths.append(filepath)

        self.filepaths = filtered_filepaths

    def _create_generators(self):
        """
        Create data generators for training and validation datasets.
        """
        train_df, val_df = self._create_dataframe(self.filepaths)

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

    def _create_dataframe(self, filepaths):
        """
        Create a dataframe from the list of filepaths.
        """
        data = {
            'filepath': filepaths,
            'class': ['PNEUMONIA' if 'PNEUMONIA' in fp else 'NORMAL' for fp in filepaths]
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
        return loss, accuracy

    def load_model(self, model_path):
        """
        Load the model from the specified path.
        """
        self.model = tf.keras.models.load_model(model_path)


if __name__ == "__main__":
    data_directory = 'datasets'
    image_size = (256, 256)
    batch_size = 32
    num_epochs = 10
    steps_per_epoch = 50
    validation_steps = 50
    test_steps = 50

    # Initialize data handlers
    data_handler = DataHandler(data_directory, image_size, batch_size)

    # Initialize the model
    detector = PneumoniaDetector(input_shape=(image_size[0], image_size[1], 3))

    # Get datasets
    train_ds = data_handler.get_dataset(data_handler.train_generator)
    val_ds = data_handler.get_dataset(data_handler.validation_generator)

    # Train the model
    history = detector.train(train_ds, val_ds, num_epochs, steps_per_epoch, validation_steps)

    # Evaluate the model
    for i in range(3):
        test_loss, test_acc = detector.evaluate(val_ds, test_steps)
        print(f'Test accuracy: {test_acc * 100:.2f}%')

    # Print image statistics
    print(f"Total images: {data_handler.image_stats['total_images']}")
    print(f"Filtered images: {data_handler.image_stats['filtered_images']}")
    print(f"Min image size: {data_handler.image_stats['min_size']}")
    print(f"Max image size: {data_handler.image_stats['max_size']}")
    print(f"Average image width: {data_handler.image_stats['avg_width']}")
    print(f"Average image height: {data_handler.image_stats['avg_height']}")
