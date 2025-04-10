# ASL Gesture Recognition with CNN and Mediapipe

This repository contains a project for real-time American Sign Language (ASL) gesture recognition using a Convolutional Neural Network (CNN) and Mediapipe Hands for hand detection and preprocessing.

## Project Overview

The goal of this project is to classify ASL gestures (letters A-Z, and special gestures: `nothing`, `space`, `del`) from live webcam input. The project uses:
- **Mediapipe Hands** for detecting hand landmarks and computing bounding boxes.
- **TensorFlow/Keras CNN model** for gesture classification.


## File Structure

- `ASL_App.ipynb`: Jupyter Notebook containing the implementation.
- `README.md`: Project documentation.

---

## Features

- **Real-Time Webcam Predictions**: Detect and classify ASL gestures in real-time from your webcam.
- **Bounding Box Visualization**: Visual feedback with bounding boxes and confidence scores for predictions.

---
## Instructions

### Environment Setup
1. **Python Version**: The project requires Python 3.10 for compatibility with MediaPipe.
2. **Create a New Environment**:
   ```bash
   conda create -n mediapipe_env python=3.10
   conda activate mediapipe_env
   ```
3. **Install Dependencies**:
   ```bash
   pip install mediapipe tensorflow opencv-python matplotlib


## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://git.txstate.edu/cac570/CS-4337.git
   cd CS-4337
   ```
2. Place the saved model (`asl_classification_model.keras`) in the root directory.
3. Activate the environment:
   ```bash
   conda activate mediapipe_env
   ```
4. Open the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
5. Run the training script or load the pre-trained model for testing.
---

## How to Run the Code

### Run with Webcam Input

To test the model with your webcam:
- Open the Jupyter Notebook or Python script containing the code.
- Run all cells.

Show ASL gestures to the webcam and observe real-time predictions!

---

## Model Details

### CNN Architecture
- **Input Shape**: 64x64x3
- **Layers**:
  - 4 Convolutional layers with ReLU activation and MaxPooling.
  - Fully connected Dense layer with 512 units.
  - Dropout for regularization.
  - Softmax output layer for 29 classes.
- **Trained on**: ASL dataset with 64x64 resized images.

---

## Future Work

- Retrain the model on higher resolution input (128x128 or 224x224) for better accuracy.
- Add more robust data augmentation to handle diverse lighting, angles, and occlusion.
- Extend to continuous gesture recognition using sequence-based models like LSTMs or Transformers.

---

