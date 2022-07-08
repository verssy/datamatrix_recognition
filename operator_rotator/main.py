
from math import acos, pi, sqrt
import cv2
import numpy as np
import pytesseract.pytesseract as pytesseract

pytesseract.tesseract_cmd=r'C:\Users\Дубровин\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

def handle_mouse(event, x, y, flags, param):
    global drawing_line_mode, config_line_points, temp_frame, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        # first click
        if drawing_line_mode == 0:
            config_line_points[0] = x
            config_line_points[1] = y
            drawing_line_mode = 1
            temp_frame = frame.copy()
            cv2.circle(temp_frame, (x, y), 3, (0, 255, 0), 2)
        else:
            config_line_points[2] = x
            config_line_points[3] = y
            drawing_line_mode = 0
            cv2.line(temp_frame, (config_line_points[0], config_line_points[1]),
                (config_line_points[2], config_line_points[3]), (0, 0, 255), 1)
            cv2.circle(temp_frame, (x, y), 3, (0, 255, 0), 2)

def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH))

drawing_line_mode = 0
frame = cv2.imread('test.jpg')
degrs = 0
temp_frame = frame.copy()
config_line_points = [-1, -1, -1, -1]
cv2.namedWindow('ColdFucktory')
cv2.setMouseCallback('ColdFucktory', handle_mouse)

while True:
    cv2.imshow('ColdFucktory', temp_frame)
    pressed_key = cv2.waitKey(1)
    if pressed_key & 0xFF == ord('r'):
        temp_frame= frame.copy()
    elif pressed_key & 0xFF == ord('q'):
        break;
    elif pressed_key == 13: # enter pressed
        if config_line_points[0] > -1 and config_line_points[2] > -1:
            vecx = config_line_points[2] - config_line_points[0]
            vecy = config_line_points[1] - config_line_points[3]
            veccos = vecx / (sqrt(vecx ** 2 + vecy ** 2))
            rads = acos(veccos) 
            degrs = int(rads * 180.0 / pi)
            if vecy < 0:
                degrs = 360 - int(degrs)
            temp_frame = rotate_bound(temp_frame, degrs)
            cv2.putText(temp_frame, 'press enter if gut', (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
            cv2.imshow('ColdFucktory', temp_frame)
            pressed_key = cv2.waitKey(0)
            if pressed_key == 13:
                break
            else:
                temp_frame = frame.copy()
                config_line_points = [-1, -1, -1, -1]
        else:
            cv2.putText(temp_frame, 'You bad', (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

custom_config = '-l eng --psm 6 --oem 3'
image_data = pytesseract.image_to_string(rotate_bound(frame, degrs), config=custom_config)
print(image_data)
