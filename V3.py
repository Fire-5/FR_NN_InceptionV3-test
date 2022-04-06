import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

<<<<<<< Updated upstream
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
=======
>>>>>>> Stashed changes
