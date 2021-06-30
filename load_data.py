import tensorflow as tf 

training_set = tf.keras.preprocessing.image_dataset_from_directory("./Dataset/train/",label_mode='binary',color_mode='grayscale')
test_set = tf.keras.preprocessing.image_dataset_from_directory("./Dataset/test/",label_mode='binary',color_mode='grayscale')