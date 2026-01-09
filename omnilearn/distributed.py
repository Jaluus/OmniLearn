import horovod.tensorflow.keras as hvd
import tensorflow as tf


def setup_gpus():
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")

