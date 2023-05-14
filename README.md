# Age-and-ageing-process-in-biometrics
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Load the dataset
data = np.load('fingerprint_data.npy')
labels = np.load('fingerprint_labels.npy')

# Split the data into training and testing sets
train_data = data[:800]
train_labels = labels[:800]
test_data = data[800:]
test_labels = labels[800:]

# Define the CNN model architecture
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(256, 256, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_absolute_error', optimizer='adam')

# Train the model
model.fit(train_data, train_labels, batch_size=32, epochs=10, verbose=1)

# Evaluate the model on the testing set
loss = model.evaluate(test_data, test_labels, verbose=0)

print('Test loss:', loss)

# Use the model to predict the age of a new fingerprint
img = cv2.imread('new_fingerprint.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))
img = np.expand_dims(img, axis=0)
img = np.expand_dims(img, axis=-1)

predicted_age = model.predict(img)

print('Predicted age:', predicted_age)
