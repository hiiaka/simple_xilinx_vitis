import os
import tensorflow as tf

def _main():

  phase = tf.compat.v1.keras.backend.learning_phase()
  print(phase)
  print("Before: Learning Phase = " + str(tf.compat.v1.keras.backend.learning_phase()))
  tf.compat.v1.keras.backend.set_learning_phase(0)
  print("After : Learning Phase = " + str(tf.compat.v1.keras.backend.learning_phase()))

  log_dir = "./outputs/"
  print(log_dir)
  model_file = log_dir + 'model/mnist.h5'
  model = tf.keras.models.load_model(model_file)
  model.summary()

  inputs = [out.op.name for out in model.inputs]
  print("inputs = ", end="")
  print(inputs)  # inputs = ['mnist_input']

  outputs = [out.op.name for out in model.outputs]
  print("outputs = ", end="")
  print(outputs)  # outputs = ['dense_1/Softmax']

  # TensorFlow形式で保存
  sess = tf.compat.v1.keras.backend.get_session()

  saver = tf.compat.v1.train.Saver()
  saver.save(sess, log_dir + "/tf_save/mnist")

if __name__ == "__main__":
  _main()
