import tensorflow as tf

def calib_input(iter):
  (train_images, train_labels), (test_images, test_labels) = \
    tf.keras.datasets.mnist.load_data()
  train_images = train_images.reshape((60000, 28, 28, 1))
  calib_images = train_images[16 * iter: 16 * (iter + 1)] / 255.0
  return {"mnist_input": calib_images}
