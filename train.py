import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the preprocessed data
train_data = tf.data.experimental.load('train_data')
test_data = tf.data.experimental.load('test_data')

# Define the data augmentation generator
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Augment the training data using ImageDataGenerator
def augment_data(dataset):
    images, labels = [], []
    for image, label in dataset:
        images.append(image.numpy())
        labels.append(label.numpy())
    images = tf.convert_to_tensor(images)
    labels = tf.convert_to_tensor(labels)
    return datagen.flow(images, labels, batch_size=32)

augmented_train_data = augment_data(train_data)

# Function to create and train the model
def create_and_train_model(base_model, train_data, test_data, epochs=6):
    base_model.trainable = False  # Freeze the base model

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(37, activation='softmax')  # 37 classes in the dataset
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(train_data, validation_data=test_data, epochs=epochs)
    return history, model

# Train MobileNetV2
base_model_mobilenet = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
history_mobilenet, model_mobilenet = create_and_train_model(base_model_mobilenet, augmented_train_data, test_data)

# Train NASNetMobile
base_model_nasnet = tf.keras.applications.NASNetMobile(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
history_nasnet, model_nasnet = create_and_train_model(base_model_nasnet, augmented_train_data, test_data)
