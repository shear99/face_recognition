import cv2
import dlib
import numpy as np
import os
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
import time
import logging

# Suppress specific macOS system messages (if on macOS)
import platform
if platform.system() == 'Darwin':
    os.environ['QT_LOGGING_RULES'] = 'qt.qpa.input=true'

# Initialize logger
logging.basicConfig(level=logging.DEBUG)

# Initialize face detector and recognizer (require shape_predictor_68_face_landmarks.dat & dlib_face_recognition_resnet_model_v1.dat)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Data storage path
data_path = "face_data"
if not os.path.exists(data_path):
    os.makedirs(data_path)

# Function to get face encodings
def get_face_encodings(image):
    logging.debug("Converting image to grayscale.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    logging.debug("Detecting faces.")
    rects = detector(gray, 1)
    encodings = []
    shapes = []
    logging.debug(f"Found {len(rects)} faces.")
    for rect in rects:
        logging.debug("Predicting facial landmarks.")
        shape = predictor(gray, rect)
        face_encoding = np.array(recognizer.compute_face_descriptor(image, shape))
        encodings.append(face_encoding)
        shapes.append(shape)
    return encodings, rects, shapes

# Function to register face
def register_face(name):
    register_window = Toplevel(root)
    register_window.title("Register Face")
    register_window.geometry("800x600")

    video_panel = Label(register_window)
    video_panel.pack(pady=10)

    timer_label = Label(register_window, text="", font=("Arial", 12))
    timer_label.pack(pady=10)

    cap = cv2.VideoCapture(0)
    registered = False
    start_time = None

    def show_frame():
        nonlocal registered, start_time
        if not video_panel.winfo_exists():
            return
        ret, frame = cap.read()
        if not ret:
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_panel.imgtk = imgtk
        video_panel.config(image=imgtk)

        encodings, rects, shapes = get_face_encodings(frame)
        if encodings and not registered:
            if start_time is None:
                start_time = time.time()
            elapsed_time = time.time() - start_time
            remaining_time = max(0, 3 - int(elapsed_time))
            timer_label.config(text=f"Saving in {remaining_time} seconds")
            if elapsed_time >= 3:
                logging.debug(f"Registering face for {name}.")
                np.save(os.path.join(data_path, f"{name}.npy"), encodings[0])
                photo_path = os.path.join(data_path, f"{name}.jpg")
                cv2.imwrite(photo_path, frame)  # Save the frame directly without conversion
                registered = True

                for rect in rects:
                    cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)
                for shape in shapes:
                    for i in range(68):
                        cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0, 0, 255), -1)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                video_panel.imgtk = imgtk
                video_panel.config(image=imgtk)
                logging.debug("Face registered successfully.")
                root.after(5000, stop_register)
            else:
                root.after(10, show_frame)
        elif not encodings:
            start_time = None
            timer_label.config(text="")
            root.after(10, show_frame)

    def stop_register():
        cap.release()
        video_panel.config(image="")
        register_window.destroy()
        update_user_list()

    show_frame()

# Function to recognize face
def recognize_face():
    recognize_window = Toplevel(root)
    recognize_window.title("Recognize Face")
    recognize_window.geometry("800x600")

    video_panel = Label(recognize_window)
    video_panel.pack(pady=10)

    cap = cv2.VideoCapture(0)

    def show_frame():
        if not video_panel.winfo_exists():
            return
        ret, frame = cap.read()
        if not ret:
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_panel.imgtk = imgtk
        video_panel.config(image=imgtk)

        encodings, _, _ = get_face_encodings(frame)
        if encodings:
            min_dist = float("inf")
            name = "Unknown"
            logging.debug("Comparing detected faces with registered faces.")
            for file in os.listdir(data_path):
                if file.endswith(".npy"):
                    data = np.load(os.path.join(data_path, file))
                    dist = np.linalg.norm(data - encodings[0])
                    logging.debug(f"Distance from {file.split('.')[0]}: {dist}")
                    if dist < min_dist:
                        min_dist = dist
                        name = file.split(".")[0]
            logging.debug(f"Minimum distance: {min_dist}")
            if min_dist < 0.6:  # Threshold for face recognition
                cv2.putText(frame, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                logging.info(f"Recognized {name} with distance {min_dist}")
            else:
                cv2.putText(frame, "Unknown", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                logging.info("Face not recognized.")
            logging.debug(f"Detected faces: {len(encodings)}")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            video_panel.imgtk = imgtk
            video_panel.config(image=imgtk)

        root.after(1000, show_frame)  # Update every second

    def stop_recognize():
        cap.release()
        video_panel.config(image="")
        recognize_window.destroy()

    show_frame()

# Function to update user list in GUI
def update_user_list():
    for widget in user_list_frame.winfo_children():
        widget.destroy()
    users = [user.split(".")[0] for user in os.listdir(data_path) if user.endswith(".npy")]
    for user in users:
        user_label = Label(user_list_frame, text=user, font=("Arial", 12))
        user_label.pack(anchor=W)
        user_label.bind("<Button-1>", lambda e, u=user: show_user_data(u))

# Function to show user data
def show_user_data(user_name):
    data = np.load(os.path.join(data_path, f"{user_name}.npy"))
    photo_path = os.path.join(data_path, f"{user_name}.jpg")
    if os.path.exists(photo_path):
        img = Image.open(photo_path)
        imgtk = ImageTk.PhotoImage(image=img)

        data_window = Toplevel(root)
        data_window.title(f"Data for {user_name}")
        data_window.geometry("800x600")

        img_label = Label(data_window, image=imgtk)
        img_label.image = imgtk  # Keep a reference to avoid garbage collection
        img_label.pack()

        text_label = Label(data_window, text=str(data), font=("Arial", 10))
        text_label.pack()

# GUI setup
def main():
    global root, user_list_frame

    root = Tk()
    root.title("Face Recognition System")
    root.geometry("800x600")

    main_frame = Frame(root)
    main_frame.pack(fill=BOTH, expand=True)

    Label(main_frame, text="Face Recognition System", font=("Arial", 20)).pack(pady=20)

    name_entry = Entry(main_frame, width=30)
    name_entry.pack(pady=10)
    name_entry.insert(0, "Enter your name")

    Button(main_frame, text="Register Face", command=lambda: threading.Thread(target=register_face, args=(name_entry.get(),)).start()).pack(pady=5)
    Button(main_frame, text="Recognize Face", command=lambda: threading.Thread(target=recognize_face).start()).pack(pady=5)

    user_list_frame = Frame(main_frame)
    user_list_frame.pack(side=RIGHT, fill=Y, padx=20, pady=20)

    Label(user_list_frame, text="Registered Users", font=("Arial", 12, "bold")).pack(anchor=W)

    update_user_list()

    root.mainloop()

if __name__ == "__main__":
    main()
