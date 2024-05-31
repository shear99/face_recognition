import cv2
import dlib
import numpy as np
import os
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk

# init(require shape_predictor_68_face_landmarks.dat & dlib_face_recognition_resnet_model_v1.dat)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# data storage path
data_path = "face_data"
if not os.path.exists(data_path):
    os.makedirs(data_path)

# 얼굴 인식 함수
def get_face_encodings(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    encodings = []
    for rect in rects:
        shape = predictor(gray, rect)
        face_encoding = np.array(recognizer.compute_face_descriptor(image, shape))
        encodings.append(face_encoding)
    return encodings

# 얼굴 등록 함수
def register_face(name):
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Register Face")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Register Face", frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC 키
            break
        elif key == ord('s'):  # 's' 키
            encodings = get_face_encodings(frame)
            if encodings:
                np.save(os.path.join(data_path, f"{name}.npy"), encodings[0])
                messagebox.showinfo("Info", f"Face for {name} registered.")
                break
    cap.release()
    cv2.destroyAllWindows()

# 얼굴 인식 함수
def recognize_face():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Recognize Face")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        encodings = get_face_encodings(frame)
        if encodings:
            for encoding in encodings:
                min_dist = float("inf")
                name = "Unknown"
                for file in os.listdir(data_path):
                    data = np.load(os.path.join(data_path, file))
                    dist = np.linalg.norm(data - encoding)
                    if dist < min_dist:
                        min_dist = dist
                        name = file.split(".")[0]
                cv2.putText(frame, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Recognize Face", frame)
        if cv2.waitKey(1) == 27:  # ESC 키
            break
    cap.release()
    cv2.destroyAllWindows()

# GUI 설정
def main():
    root = Tk()
    root.title("Face Recognition System")
    root.geometry("400x200")

    Label(root, text="Face Recognition System", font=("Arial", 20)).pack(pady=20)

    name_entry = Entry(root, width=30)
    name_entry.pack(pady=10)
    name_entry.insert(0, "Enter your name")

    Button(root, text="Register Face", command=lambda: register_face(name_entry.get())).pack(pady=5)
    Button(root, text="Recognize Face", command=recognize_face).pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    main()
