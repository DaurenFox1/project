import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
known_image = cv2.imread("photo1673784268.jpeg")
known_gray = cv2.cvtColor(known_image, cv2.COLOR_BGR2GRAY)

known_faces = face_cascade.detectMultiScale(known_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

if len(known_faces) == 0:
    print("No faces found in the known image.")
else:
    x, y, w, h = known_faces[0]
    known_face_region = known_gray[y:y + h, x:x + w]
    orb = cv2.ORB_create()
    known_keypoints, known_descriptors = orb.detectAndCompute(known_face_region, None)
    known_face_name = "Dauren"

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_region = gray_frame[y:y + h, x:x + w]
        keypoints, descriptors = orb.detectAndCompute(face_region, None)

        if descriptors is not None and known_descriptors is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(known_descriptors, descriptors)

            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) > 10:
                name = known_face_name
            else:
                name = "Unknown"
        else:
            name = "Unknown"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow('Project For Computer Vision', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

