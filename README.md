# Fruit Classification with MobileNetV2 and Real-Time Webcam Detection
This project is a fruit classification system built using a pre-trained MobileNetV2 model. It allows users to classify fruits from an image dataset and detect fruits in real-time using a webcam.

## Features
- Train a Fruit Classification Model: Uses MobileNetV2 with transfer learning for accurate classification.
- Real-Time Fruit Detection: Utilizes a webcam to predict fruit types in real-time.
- 10 Fruit Classes: Includes apples, bananas, mangoes, watermelons, and more.
- Data Augmentation: Enhances the training dataset with transformations like rotation, zoom, and flipping.

## Installation
1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/your-username/fruit-classification.git
    cd fruit-classification

2. Install the required dependencies:
   ```bash
   pip install tensorflow numpy matplotlib opencv-python
3. Prepare your dataset:
   - Organize your images into `train` and `test` directories.
   - Example structure:
     ```bash
     Data Science Projects/
      ├── Fruit_Vegie/
          ├── train/
              ├── apple/
              ├── banana/
              ├── mango/
              └── ...
          └── test/
              ├── apple/
              ├── banana/
              ├── mango/
              └── ...

 # How to Run
 ## Training the Model
 1. Place your dataset in the directory specified in the code.
 2. Run the training script:
    ```bash
    python script_name.py
3. The trained model will be saved as `fruit_classifier_model.h5.`

   ## Real-Time Fruit Detection
   1. Ensure you have a webcam connected to your system.
   2. Run the real-time detection:
      ```bash
      python script_name.py
3. Point your webcam at a fruit to see the prediction displayed on the video feed.
4. Press 'x' to exit the real-time detection mode.

## Fruit Classes
The model is trained to classify the following fruits:

1. Apple
2. Banana
3. Durian
4. Jackfruit
5. Mango
6. Orange
7. Pineapple
8. Pomegranate
9. Tomato
10. Watermelon

## Results
- Test Accuracy: Achieved during training and validation (check the terminal output).
- Real-Time Prediction: The detected fruit label is displayed on the webcam feed.

## Dependencies
- Python 3.7 or later
- TensorFlow
- NumPy
- Matplotlib
- OpenCV

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
