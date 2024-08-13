Project 1: Real-Time Hand Gesture Recognition (app.ipynb)
Description:
This project focuses on real-time hand gesture recognition using the MediaPipe library for detecting and processing hand landmarks. The system captures hand movements through a webcam and interprets them as specific gestures. It utilizes OpenCV for image processing and visualization, alongside TensorFlow to load and use pre-trained models.

Key Features:

Real-Time Detection: Utilizes MediaPipe to perform real-time hand detection and landmark recognition.
Gesture Recognition: Recognizes predefined hand gestures by processing the detected hand landmarks.
Integration with TensorFlow: The project uses a TensorFlow model for gesture classification, allowing for efficient and accurate predictions.
User-Friendly Interface: Visual feedback is provided through the webcam feed, displaying detected gestures in real-time.
Technologies Used:

Python
OpenCV
MediaPipe
TensorFlow
NumPy
Project 2: Gesture Image Collection for Training (gestures.ipynb)
Description:
This project is designed to collect and organize a dataset of hand gesture images for training machine learning models. It captures images from a webcam and sorts them into directories corresponding to specific gestures. This dataset can be used for training models in gesture recognition tasks.

Key Features:

Dataset Creation: Automatically captures images from a webcam and categorizes them into directories based on the gesture being performed.
Gesture Labels: Supports a variety of common gestures such as 'Hello', 'Yes', 'No', 'Thank you', and more.
Interactive Image Capture: The user can control the image capture process via keyboard inputs, allowing for easy collection of gesture data.
Technologies Used:

Python
OpenCV
NumPy
OS module


I used my own dataset to train both the models
