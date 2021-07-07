import tensorflow as tf 
import numpy as np

image_shape = (224,224)
input_shape = image_shape+(3,)
means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]
variances = [0.229*0.229, 0.224*0.224, 0.225*0.225]

training_set = tf.keras.preprocessing.image_dataset_from_directory("./Dataset/train/",image_size=image_shape,label_mode='binary',color_mode='rgb')#color_mode='grayscale')
test_set = tf.keras.preprocessing.image_dataset_from_directory("./Dataset/valid/",image_size=image_shape,label_mode='binary',color_mode='rgb')#color_mode='grayscale')


# https://keras.io/guides/transfer_learning/

model = tf.keras.applications.VGG16(weights='imagenet',input_shape=input_shape,include_top=False)
model.trainable = False

inputs = tf.keras.Input(shape=input_shape)
x = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./255)(inputs)
x = tf.keras.layers.experimental.preprocessing.Normalization(mean=means,variance=variances)(x)

x =  model(x,training=False)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(4096,activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(4096,activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(1,activation="sigmoid")(x)



model = tf.keras.Model(inputs,outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

model.summary()

model.fit(training_set,epochs=2)