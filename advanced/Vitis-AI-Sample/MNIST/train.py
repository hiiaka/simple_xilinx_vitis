import tensorflow as tf
from model.model import create_model

def _main():

  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
  train_images = train_images.reshape((60000, 28, 28, 1))
  test_images = test_images.reshape((10000, 28, 28, 1))
  train_images, test_images = train_images / 255.0, test_images / 255.0

  model = create_model()
  model.summary()

  model.fit(train_images, train_labels, epochs=5)

  print("Evaluation")
  test_loss, test_acc = model.evaluate(test_images, test_labels)
  print("Accuracy = " + str(test_acc))

  # モデル保存
  log_dir = './outputs/model/'
  model.save(log_dir + "mnist.h5")

if __name__ == "__main__":
  _main()
