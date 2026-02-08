import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPUs detected: {tf.config.list_physical_devices('GPU')}")
