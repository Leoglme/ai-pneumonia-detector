import sys
import os

# Hide TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.device_manager import DeviceManager

if __name__ == "__main__":
    # Compare performance between CPU and GPU
    device_manager = DeviceManager()
    device_manager.use_best_device()

    # Train models
