# --- TensorFlow ---

import tensorflow as tf

print(f"[DEBUG] TensorFlow version {tf.__version__}")

devices = tf.config.list_physical_devices("GPU")
if len(devices) > 0:
    rnd = tf.random.uniform(shape=(100, 1))
    print("[STATUS] GPU available")
else:
    print("[STATUS] GPU not available")
