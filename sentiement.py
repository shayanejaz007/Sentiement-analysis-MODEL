import cv2
import numpy as np
from keras.models import load_model
import time
import threading
import music
import pygame

# Function to play music in a separate thread


def play_music_thread(file_path):
    music.play_song(file_path)


# Load the pre-trained model
model = load_model(
    r'C:\Users\koks\Desktop\jarvis\model_file_100epochs.h5')

# Open the webcam
video = cv2.VideoCapture(0)
faceDetect = cv2.CascadeClassifier(
    r'C:\Users\koks\Desktop\jarvis\haarcascade_frontalface_default.xml')

labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear',
               3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Flag to control music playback
play_music_flag = False
current_song_path = None

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)

    for (x, y, w, h) in faces:
        sub_face_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized/255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, labels_dict[label], (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        print(f"predicted result is {labels_dict[label]}")

        # If Happy emotion is detected, start playing happy music
        if label == 3 and not play_music_flag and current_song_path != r'C:\Users\koks\Desktop\jarvis\happy songs\happy.mp3':
            play_music_flag = True
            current_song_path = r'C:\Users\koks\Desktop\jarvis\happy songs\happy.mp3'
            music_thread = threading.Thread(
                target=play_music_thread, args=(current_song_path,))
            music_thread.start()

        # If Sad emotion is detected, start playing sad music
        elif label == 5 and not play_music_flag and current_song_path != r'C:\Users\koks\Desktop\jarvis\sad songs\sad1.mp3':
            play_music_flag = True
            current_song_path = r'C:\Users\koks\Desktop\jarvis\sad songs\sad1.mp3'
            music_thread = threading.Thread(
                target=play_music_thread, args=(current_song_path,))
            music_thread.start()

        # If Neutral emotion is detected, stop playing any music
        elif label == 4 and play_music_flag:
            play_music_flag = False
            # Stop the current song
            pygame.mixer.music.stop()

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()
