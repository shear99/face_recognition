# Face Recognition System

This system allows users to register their faces and recognize registered faces using a webcam. It is built using Python, OpenCV, dlib, and Tkinter for the GUI.

## Requirements

- Python 3.x
- OpenCV
- dlib
- numpy
- PIL (Pillow)
- tkinter

## Setup

1. Install the required libraries:
    ```bash
    pip install opencv-python dlib numpy pillow
    ```

2. Download the required dlib models and place them in the same directory as the script:
    - `shape_predictor_68_face_landmarks.dat`
    - `dlib_face_recognition_resnet_model_v1.dat`

## Usage

1. Run the script:
    ```bash
    python face_recognition_system.py
    ```

2. The GUI will open with options to register a new face or recognize faces.

### Registering a Face

1. Enter your name in the text box.
2. Click "Register Face".
3. The system will start the webcam and capture your face. Hold still for 3 seconds to complete the registration.
4. The face encoding and image will be saved in the `face_data` directory.

### Recognizing Faces

1. Click "Recognize Face".
2. The system will start the webcam and attempt to recognize any registered faces.
3. Recognized faces will be displayed with the name, and the result will be printed to the terminal.
