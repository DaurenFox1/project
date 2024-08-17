import cv2
import os

# Constants
FRAME_WIDTH = 1000
FRAME_HEIGHT = 480
MIN_AREA = 500
CASCADE_PATH = "justProject/haarcascade_russian_plate_number.xml"
IMAGE_SAVE_PATH = "justProject/img/"
BRIGHTNESS = 150

count = 0

plate_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if plate_cascade.empty():
    print("Error loading cascade file")
    exit()

cap = cv2.VideoCapture(0)
cap.set(3, FRAME_WIDTH)
cap.set(4, FRAME_HEIGHT)
cap.set(10, BRIGHTNESS)

while True:
    success, img = cap.read()
    if not success:
        break

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    number_plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    for (x, y, w, h) in number_plates:
        area = w * h
        if area > MIN_AREA:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            img_roi = img[y:y + h, x:x + w]
            cv2.imshow("Number Plate", img_roi)

    cv2.imshow("Result", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        try:
            if not os.path.exists(IMAGE_SAVE_PATH):
                os.makedirs(IMAGE_SAVE_PATH)
            cv2.imwrite(f"{IMAGE_SAVE_PATH}{str(count)}.jpg", img_roi)
            cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "Scan Saved", (15, 265), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
            cv2.imshow("Result", img)
            cv2.waitKey(500)
            count += 1
        except NameError:
            print("No number plate detected to save.")
    elif key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
