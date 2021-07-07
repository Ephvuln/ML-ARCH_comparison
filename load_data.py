import tensorflow as tf 
import numpy as np

image_shape = (224,224)
input_shape = image_shape+(3,)


training_set = tf.keras.preprocessing.image_dataset_from_directory("./Dataset/train/",image_size=image_shape,label_mode='binary',color_mode='rgb')#color_mode='grayscale')
test_set = tf.keras.preprocessing.image_dataset_from_directory("./Dataset/valid/",image_size=image_shape,label_mode='binary',color_mode='rgb')#color_mode='grayscale')


# https://keras.io/guides/transfer_learning/
model = tf.keras.applications.VGG16(weights='imagenet',input_shape=input_shape,include_top=False)

inputs = tf.keras.Input(shape=input_shape)
x =  model(inputs,training=False)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(4096,activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(4096,activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(1,activation="sigmoid")(x)



model = tf.keras.Model(inputs,outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

model.summary()

model.fit(training_set,epochs=2)