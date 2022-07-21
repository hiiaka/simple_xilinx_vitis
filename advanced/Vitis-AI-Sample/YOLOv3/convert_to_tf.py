import os
import tensorflow as tf

def _main():

  phase = tf.compat.v1.keras.backend.learning_phase()
  print (phase)
  print ("Before: Learning Phase = " + str(tf.compat.v1.keras.backend.learning_phase()))
  tf.compat.v1.keras.backend.set_learning_phase(0)
  print ("After : Learning Phase = " + str(tf.compat.v1.keras.backend.learning_phase()))

  log_dir = "./outputs/"
  print(log_dir)
  model = tf.keras.models.load_model("yolo.h5", compile=False)
  model.summary()

  inputs = [out.op.name for out in model.inputs]
  print("inputs = ", end="")
  print(inputs)  # inputs = ['input_1']

  outputs = [out.op.name for out in model.outputs]
  print("outputs = ", end="")
  print(outputs)  # outputs = ['conv2d_59/BiasAdd', 'conv2d_67/BiasAdd', 'conv2d_75/BiasAdd']

  # TensorFlow形式で保存
  sess = tf.compat.v1.keras.backend.get_session()

  saver = tf.compat.v1.train.Saver()
  saver.save(sess, log_dir + "/model/yolov3")

if __name__ == "__main__":
  _main()
