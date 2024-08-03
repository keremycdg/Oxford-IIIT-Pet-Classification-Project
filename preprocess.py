import tensorflow as tf
import tensorflow_datasets as tfds

# Load the Oxford-IIIT Pet Dataset
dataset, info = tfds.load('oxford_iiit_pet', with_info=True, as_supervised=True)
print(info)

# Function to preprocess images
def preprocess_image(image, label):
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    return image, label

# Apply preprocessing
train_data = dataset['train'].map(preprocess_image).batch(32).prefetch(tf.data.AUTOTUNE)
test_data = dataset['test'].map(preprocess_image).batch(32).prefetch(tf.data.AUTOTUNE)

# Save preprocessed data
tf.data.experimental.save(train_data, 'train_data')
tf.data.experimental.save(test_data, 'test_data')
