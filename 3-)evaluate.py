import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

# Load the preprocessed data
test_data = tf.data.experimental.load('test_data')

# Function to convert tf.data.Dataset to numpy arrays
def dataset_to_numpy(dataset):
    images, labels = [], []
    for image, label in dataset:
        images.append(image.numpy())
        labels.append(label.numpy())
    return np.stack(images), np.stack(labels)

test_images, test_labels = dataset_to_numpy(test_data)

# Load trained models
model_mobilenet = tf.keras.models.load_model('model_mobilenet.h5')
model_nasnet = tf.keras.models.load_model('model_nasnet.h5')

# Predict on the test data
test_predictions_mobilenet = model_mobilenet.predict(test_images)
test_predictions_classes_mobilenet = np.argmax(test_predictions_mobilenet, axis=1)

test_predictions_nasnet = model_nasnet.predict(test_images)
test_predictions_classes_nasnet = np.argmax(test_predictions_nasnet, axis=1)

# Function to plot random actual vs predicted classes
def plot_random_actual_vs_predicted(images, actual, predicted, class_names, num=10):
    indices = random.sample(range(len(images)), num)
    plt.figure(figsize=(20, 10))
    for i, idx in enumerate(indices):
        plt.subplot(2, num // 2, i + 1)
        plt.imshow(images[idx])
        plt.title(f'Actual: {class_names(actual[idx])}\nPredicted: {class_names(predicted[idx])}')
        plt.axis('off')
    plt.show()

# Get class names
dataset, info = tfds.load('oxford_iiit_pet', with_info=True, as_supervised=True)
class_names = info.features['label'].int2str

# Show random actual and predicted classes
plot_random_actual_vs_predicted(test_images, test_labels, test_predictions_classes_mobilenet, class_names, num=10)
plot_random_actual_vs_predicted(test_images, test_labels, test_predictions_classes_nasnet, class_names, num=10)
