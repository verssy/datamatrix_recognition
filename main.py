
from math import acos, pi, sqrt
from time import sleep
import cv2
import pytesseract.pytesseract as pytess
import pylibdmtx.pylibdmtx as pdmtx
from colorama import Fore
import pyfiglet
import rich
import numpy as np
import re

pytess.tesseract_cmd=r'C:\Users\Дубровин\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# logo
title = pyfiglet.figlet_format('Cold Factory', font='invita')
rich.print(f'[spring_green3]{title}[/spring_green3]')

# args
files_prefix = "./imgs/barcode_"
max_sides_ratio = 1.5
min_area = 600
delta_xy = 10
bool_save_video = True
cap_aspect = 3
input_filename = 'vid_1.mp4'
images = []
custom_config = '-l deu --psm 6 --oem 3'

if input_filename.count('.jpg') or input_filename.count('.jpeg') or input_filename.count('.png'):
    bool_video = False
else:
    bool_video = True

# found degrees
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

degrs = 0
drawing_line_mode = 0
config_line_points = [-1, -1, -1, -1]
cv2.namedWindow('ColdFucktory')
cv2.setMouseCallback('ColdFucktory', handle_mouse)
frame = None
if bool_video:
    cap = cv2.VideoCapture(input_filename)
    while True:
        _, frame = cap.read()
        if not _: break
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        temp_frame = frame.copy()
        cv2.putText(temp_frame, 'g - good frame', (5, 35), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 2)
        cv2.putText(temp_frame, 'n - next frame', (5, 75), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 2)
        cv2.imshow('ColdFucktory', temp_frame)
        pressed_key = 0
        while (pressed_key & 0xFF) not in [ord('g'), ord('n'), ord('q')]:
            pressed_key = cv2.waitKey(0)
        if pressed_key & 0xFF == ord('g'): break
        elif pressed_key & 0xFF == ord('q'): exit(0)
else:
    frame = cv2.imread(input_filename)

temp_frame = frame.copy()
while True:
    cv2.putText(temp_frame, 'draw line (2 clicks)', (5, 35), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 2)
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
            temp_temp_frame = rotate_bound(temp_frame, degrs)
            cv2.putText(temp_temp_frame, 'press enter if gut', (10, 75), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 2)
            cv2.imshow('ColdFucktory', temp_temp_frame)
            pressed_key = cv2.waitKey(0)
            if pressed_key == 13:
                break
            else:
                temp_frame = frame.copy()
                config_line_points = [-1, -1, -1, -1]
        else:
            cv2.putText(temp_frame, 'You bad', (10, 75), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 2)
cv2.destroyAllWindows()
print(f'rotation in degrees: {degrs}')

# main part
if bool_video: cap = cv2.VideoCapture(input_filename)
while True:
    if bool_video:
        _, frame = cap.read()
        if not _: break
    else: frame = cv2.imread(f'{input_filename}')

    temp_frame = frame.copy()
    temp_frame = rotate_bound(temp_frame, degrs)
    image_data = pytess.image_to_string(temp_frame, config=custom_config)
    print(image_data)
    m = re.findall('[0-9][0-9][.,\/-][0-9][0-9][.,\/-][0-9][0-9]', image_data)

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, None, fx=1/cap_aspect, fy=1/cap_aspect, interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
    harris = cv2.cornerHarris(img_gray, 4, 3, 0.06)
    __, thr = cv2.threshold(harris, 0.001 * harris.max(), 255, cv2.THRESH_BINARY)
    thr = thr.astype('uint8')

    contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = list(filter(lambda x: cv2.contourArea(cv2.convexHull(x)) > min_area, contours))

    for con in contours:
        (x, y, w, h) = cv2.boundingRect(con)
        dx = min(x, 5)
        dy = min(y, 5)
        x = x - dx
        y = y - dy
        h = h + dy * 2
        w = w + dx * 2
        if not (w / h > max_sides_ratio or h / w > max_sides_ratio):
            tempimage = frame[int(y * cap_aspect): int(y * cap_aspect + h * cap_aspect), int(x * cap_aspect): int(x * cap_aspect + w * cap_aspect)]
            data = pdmtx.decode(tempimage, max_count=1)
            text = ""
            if len(data) > 0:
                text = data[0].data.decode('UTF-8')
                text = text.replace('\x1d', '\\x1d')
                spl = text.split('\\x1d')
                if len(spl) == 2:
                    print(Fore.GREEN + spl[0] + Fore.CYAN + '\\1xd' + Fore.GREEN + spl[1] + Fore.RESET)
                else:
                    print(f"found barcode: {text}")
                
                cv2.putText(frame, text, (int(x * cap_aspect), int(y * cap_aspect + h * cap_aspect + 20)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (int(x * cap_aspect), int(y * cap_aspect)), (int(x * cap_aspect + w * cap_aspect), int(y * cap_aspect + h * cap_aspect)), (0, 255, 0))
    
    if len(m) > 0:
        str_m = str(m)
        str_m = str_m.replace('/', '.')
        str_m = str_m.replace(',', '.')
        str_m = str_m.replace('-', '.')
        cv2.putText(frame, 'date read: ' + str_m, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    cv2.imshow("image", cv2.resize(frame, (0, 0), fx=0.5, fy=0.5))
    if bool_video and bool_save_video: images.append(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('I quit!')
        break

print('complite!')

if bool_video:
    cap.release()
    if bool_save_video and len(images) > 0:
        (height, width, layers) = images[1].shape
        video = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))
        for i in range(len(images)):
            video.write(images[i])
        video.release()

cv2.destroyAllWindows()
