import cv2
import os
import numpy as np

# Загрузка каскадов для обнаружения лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Путь к известным изображениям
known_image_paths = ["photo1716056802.jpeg", "photo1715448229.jpeg"]
known_faces_data = []

# Загрузка и обработка известных изображений
for image_path in known_image_paths:
    known_image = cv2.imread(image_path)
    if known_image is None:
        print(f"Error loading image: {image_path}")
        continue
    known_gray = cv2.cvtColor(known_image, cv2.COLOR_BGR2GRAY)
    known_faces = face_cascade.detectMultiScale(known_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(known_faces) == 0:
        print(f"No faces found in the known image: {image_path}")
        continue

    x, y, w, h = known_faces[0]
    known_face_region = known_gray[y:y + h, x:x + w]
    orb = cv2.ORB_create()
    known_keypoints, known_descriptors = orb.detectAndCompute(known_face_region, None)
    known_faces_data.append({
        "name": os.path.splitext(os.path.basename(image_path))[0],
        "descriptors": known_descriptors
    })

# Инициализация захвата видео
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_region = gray_frame[y:y + h, x:x + w]
        keypoints, descriptors = orb.detectAndCompute(face_region, None)

        name = "Unknown"
        color = (0, 0, 255)
        best_match_count = 0

        if descriptors is not None:
            for known_face in known_faces_data:
                known_descriptors = known_face["descriptors"]
                if known_descriptors is not None:
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = bf.match(known_descriptors, descriptors)
                    matches = sorted(matches, key=lambda x: x.distance)

                    if len(matches) > best_match_count and len(matches) > 50:  # Увеличение порога совпадений
                        best_match_count = len(matches)
                        name = known_face["name"]
                        color = (0, 255, 0)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Project For Computer Vision', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

