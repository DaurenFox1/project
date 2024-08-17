import cv2

frameWidth = 1000
frameHeight = 480


plateCascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
minArea = 500

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

count = 0

while True:
    success, img = cap.read()
    if not success:
        break
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    numberPlates = plateCascade.detectMultiScale(imgGray, 1.1, 4)


    for (x, y, w, h) in numberPlates:
        area = w * h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 2)
            imgRegionOfInterest = img[y:y + h, x:x + w]
            cv2.imshow("Number Plate", imgRegionOfInterest)
    cv2.imshow("Result", img)


    if cv2.waitKey(1) & 0xFF == ord('s'):
        try:
            #When you put the buttom "s", program should the take a photo interest region(number plate of car)
            cv2.imwrite(f"./img/{str(count)}.jpg", imgRegionOfInterest)
            cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "Scan Saved", (15, 265), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Result", img)
            cv2.waitKey(500)
            count += 1

        except NameError:
            print("No number plate detected to save.")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
