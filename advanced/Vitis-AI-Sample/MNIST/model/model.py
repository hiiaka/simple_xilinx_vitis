import tensorflow as tf

def mnist_body():

  inputs = tf.keras.layers.Input(shape=(28, 28, 1), name="mnist_input")
  x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
  x = tf.keras.layers.MaxPooling2D((2, 2))(x)
  x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
  x = tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1), padding='same')(x)
  x = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(64, activation='relu')(x)
  x = tf.keras.layers.Dense(10, activation='softmax')(x)

  model = tf.keras.models.Model(inputs, x, name="MNIST_model")

  return model

def create_model():

  model = mnist_body()
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  return model

if __name__ == "__main__":
  model = create_model()
  model.summary()