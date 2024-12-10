# Importing necessary libraries
import os  # For handling file and directory operations
import numpy as np  # For numerical operations, especially array manipulation
import matplotlib.pyplot as plt  # For plotting graphs (accuracy and loss)
import tensorflow as tf  # For building and training the neural network
import cv2  # For working with computer vision (webcam input, image processing)
from tensorflow.keras.models import Sequential  # For creating a sequential neural network model
from tensorflow.keras.layers import Dense, Flatten, Dropout  # Layers used in the neural network
from tensorflow.keras.applications import MobileNetV2  # Pretrained MobileNetV2 for transfer learning
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img  # Image preprocessing tools

# Data augmentation for training to help the model generalize better
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,           # Normalize pixel values to the range [0, 1]
    rotation_range=20,           # Randomly rotate images within a range of 20 degrees
    width_shift_range=0.2,       # Randomly shift the image horizontally (up to 20%)
    height_shift_range=0.2,      # Randomly shift the image vertically (up to 20%)
    shear_range=0.2,             # Apply shear transformation (slanting the image)
    zoom_range=0.2,              # Apply random zoom on the image
    horizontal_flip=True         # Randomly flip the image horizontally to increase dataset diversity
)

# Data augmentation for testing, with only rescaling (no transformations)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load the training data using flow_from_directory
train_data = train_datagen.flow_from_directory(
    r"...\Data Science Projects\Fruit_Vegie\train",  # Path to training data
    target_size=(224, 224),      # Resize images to match the MobileNetV2 input size (224x224)
    batch_size=32,               # Number of images processed per batch
    class_mode='categorical',    # Multi-class classification (labels are one-hot encoded)
    color_mode='rgb'             # Use RGB color images
)

# Load the testing data similarly
test_data = test_datagen.flow_from_directory(
    r"...\Data Science Projects\Fruit_Vegie\test",  # Path to test data
    target_size=(224, 224),      # Resize images for testing
    batch_size=32,               # Number of images per batch during testing
    class_mode='categorical',    # Multi-class classification
    color_mode='rgb'             # Use RGB color images
)

# Load the MobileNetV2 model with pre-trained weights from ImageNet (excluding the top layer)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the pre-trained layers to prevent them from being updated during training

# Build a custom model by adding layers on top of the pre-trained MobileNetV2 base model
model = Sequential([
    base_model,                   # Add MobileNetV2 as the base model
    Flatten(),                    # Flatten the 2D output of MobileNetV2 to 1D
    Dropout(0.25),                # Add dropout layer to prevent overfitting (25% dropout)
    Dense(128, activation='relu'), # Fully connected layer with 128 neurons and ReLU activation
    Dense(10, activation='softmax') # Output layer with 10 classes (softmax for multi-class classification)
])

# Compile the model using the Adam optimizer and categorical crossentropy loss function
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using the training data, and validate it with the test data
history = model.fit(
    train_data,                   # Training data
    epochs=15,                    # Number of epochs for training
    validation_data=test_data     # Validation data (testing data)
)

# Evaluate the trained model on the test dataset and print the test accuracy
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Save the trained model to a file for future use (e.g., model deployment)
model.save('fruit_classifier_model.h5')

# Fruit labels corresponding to the class indices (ensure these match the directory structure)
labels = ['apple', 'banana', 'durian', 'jackfruit', 'mango', 
          'orange', 'pineapple', 'pomegranate', 'tomato', 'watermelon']

# Open the webcam to capture real-time images for fruit prediction
cap = cv2.VideoCapture(0)  # 0 is the default camera ID

# Start an infinite loop to capture frames from the webcam
while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:  # Break the loop if the camera is not working properly
        break

    # Preprocess the captured frame to match the input shape required by the model
    img = cv2.resize(frame, (224, 224))  # Resize the frame to 224x224
    img = img_to_array(img) / 255.0      # Convert the frame to a numpy array and normalize the pixel values
    img = np.expand_dims(img, axis=0)    # Add an extra batch dimension (as the model expects batches)

    # Predict the class of the current frame using the trained model
    predictions = model.predict(img)
    class_index = np.argmax(predictions) # Get the index of the class with the highest predicted probability
    label = labels[class_index]          # Map the class index to its corresponding label

    # Display the predicted label on the captured frame
    cv2.putText(frame, f"It matches with: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the captured frame with the prediction on the screen
    cv2.imshow("Fruit Detector by Safwan & Arsyi", frame)

    # Exit the loop when the 'x' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# Release the webcam and close any OpenCV windows after the loop
cap.release()
cv2.destroyAllWindows()
