import cv2
import numpy as np
from keras.models import load_model
import urllib.request
import time
URL = 'http://192.168.68.104:8080/video'
def nothing(x):
    pass
image_x, image_y = 64,64
classifier = load_model('model.h5')
classes = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7,
    'i': 8,
    'j': 9,
    'k': 10,
    'l': 11,
    'm': 12,
    'n': 13,
    'o': 14,
    'p': 15,
    'q': 16,
    'r': 17,
    's': 18,
    't': 19,
    'u': 20,
    'v': 21,
    'w': 22,
    'x': 23,
    'y': 24,
    'z': 25,
}
def predictor():
    from keras.preprocessing import image
    test_image = image.load_img('1.png', target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.array(test_image, dtype='float32')
    test_image = test_image/255
    test_image = np.expand_dims(test_image, axis = 0)
    predictions = classifier.predict_classes(test_image)
    print(predictions)
    key = next(key for key, value in classes.items() if value == predictions[0])
    return key

       

cam = cv2.VideoCapture(0)
cap = cv2.VideoCapture()
cap.open("http://192.168.68.104:8080/video")
cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)
cv2.namedWindow("test")

img_counter = 0

img_text = ''
x_crop_prev, y_crop_prev, w_crop_prev, h_crop_prev = 0, 0, 0, 0



prevcnt = np.array([], dtype=np.int32)

gestureStatic = 0
while True:
    ret, frame = cap.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100)
    min_HSV = np.array([cv2.getTrackbarPos('L - H', 'Trackbars'),
                        cv2.getTrackbarPos('L - S', 'Trackbars'),
                        cv2.getTrackbarPos('L - V', 'Trackbars')], np.uint8)
    max_HSV = np.array([cv2.getTrackbarPos('U - H', 'Trackbars'),
                        cv2.getTrackbarPos('U - S', 'Trackbars'),
                        cv2.getTrackbarPos('U - V', 'Trackbars')], np.uint8)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinRegion = cv2.inRange(hsv, min_HSV, max_HSV)
    contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cnt = contours[0]
    ret = cv2.matchShapes(cnt, prevcnt, 2, 0.0)
    prevcnt = contours[0]
    if (ret > 0.70):
        gestureStatic = 0
    else:
        gestureStatic += 1
    x_crop, y_crop, w_crop, h_crop = cv2.boundingRect(cnt)



    if (abs(x_crop - x_crop_prev) > 50 or abs(y_crop - y_crop_prev) > 50 or
            abs(w_crop - w_crop_prev) > 50 or abs(h_crop - h_crop_prev) > 50):
        x_crop_prev = x_crop
        y_crop_prev = y_crop
        h_crop_prev = h_crop
        w_crop_prev = w_crop

    mask = frame.copy()[max(0, y_crop_prev - 50):y_crop_prev + h_crop_prev + 50,
                max(0, x_crop_prev - 50):x_crop_prev + w_crop_prev + 50]
    cv2.rectangle(frame, (x_crop, y_crop), (x_crop + w_crop, y_crop + h_crop), (225, 0, 0), 2)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.GaussianBlur(mask, (41, 41), 0)
    ret, mask = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY_INV)

    if gestureStatic == 10:
        print("Gesture Detected")
        img_name = "1.png"
        save_img = cv2.resize(mask, (image_x, image_y))
        cv2.imwrite(img_name, save_img)
        print("{} written!".format(img_name))
        img_text = predictor()

    cv2.putText(frame, img_text, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
    cv2.imshow("test", frame)
    cv2.imshow("mask", mask)
    


        

    if cv2.waitKey(1) == 27:
        break


cam.release()
cv2.destroyAllWindows()