import os
# Disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf


class DataHandler:
    """
    This class handles the creation of data generators for training, validation, and testing datasets.
    """

    def __init__(self, data_dir, img_size=(235, 235), batch_sz=50):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_sz = batch_sz
        self.train_dir = os.path.join(data_dir, 'train')
        self.validation_dir = os.path.join(data_dir, 'val')
        self.test_dir = os.path.join(data_dir, 'test')
        self.train_generator = None
        self.validation_generator = None
        self.test_generator = None
        self._create_generators()

    def _create_generators(self):
        """
        Create data generators for training, validation, and testing datasets.
        """
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

        self.train_generator = datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=self.batch_sz,
            class_mode='binary'
        )

        self.validation_generator = datagen.flow_from_directory(
            self.validation_dir,
            target_size=self.img_size,
            batch_size=self.batch_sz,
            class_mode='binary'
        )

        self.test_generator = datagen.flow_from_directory(
            self.test_dir,
            target_size=self.img_size,
            batch_size=self.batch_sz,
            class_mode='binary'
        )


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
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, train_gen, val_gen, epochs=5, train_steps=10, val_steps=50):
        """
        Train the model using the training and validation data generators.
        """
        history = self.model.fit(
            train_gen,
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=val_steps
        )
        return history

    def evaluate(self, test_gen, steps):
        """
        Evaluate the model using the test data generator.
        """
        loss, accuracy = self.model.evaluate(test_gen, steps=steps)
        return loss, accuracy


if __name__ == "__main__":
    data_directory = 'datasets/chest_Xray'
    image_size = (128, 128)
    batch_size = 50
    num_epochs = 5
    steps_per_epoch = 10
    validation_steps = 50
    test_steps = 50

    # Initialize data handlers
    data_handler = DataHandler(data_directory, image_size, batch_size)

    # Initialize the model
    detector = PneumoniaDetector(input_shape=(image_size[0], image_size[1], 3))

    # Train the model
    detector.train(data_handler.train_generator, data_handler.validation_generator, num_epochs, steps_per_epoch, validation_steps)

    # Evaluate the model
    test_loss, test_acc = detector.evaluate(data_handler.test_generator, test_steps)
    print(f'Test accuracy: {test_acc * 100:.2f}%')
