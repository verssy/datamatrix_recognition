
import os
import cv2
import pytesseract
import pylibdmtx.pylibdmtx as pdmtx
import easyocr

#pytesseract.pytesseract.tesseract_cmd=r'C:\Users\Дубровин\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# args
files_prefix = "./imgs/barcode_"
max_sides_ratio = 1.5
min_area = 1000
delta_xy = 10
bool_save_video = True
bool_video = True
bool_fast_decode = False

def is_probably_date_string(inputString):
    return any(char.isdigit() for char in inputString) and len(inputString)>5

text_reader = easyocr.Reader(['ru'], gpu=True)
files_count, images = 0, []

if bool_video: cap = cv2.VideoCapture('./vid_1.mp4')

while True:
    if bool_video:
        _, frame = cap.read()
        if not _: break
    else: frame = cv2.imread('test1.jpg')

    # text recognition
    recognized = text_reader.readtext(frame, rotation_info=[180])
    for (bbox, text, prob) in recognized:
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        tr = (int(tr[0]), int(tr[1]))
        br = (int(br[0]), int(br[1]))
        bl = (int(bl[0]), int(bl[1]))
        if is_probably_date_string(text):
            cv2.rectangle(frame, tl, br, (0, 255, 0), 1)
            cv2.putText(frame, text, (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
    
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
    harris = cv2.cornerHarris(img_gray, 4, 3, 0.06)
    __, thr = cv2.threshold(harris, 0.001 * harris.max(), 255, cv2.THRESH_BINARY)
    thr = thr.astype('uint8')

    #d = pytesseract.image_to_data(frame, output_type=pytesseract.Output.DICT)
    #n_boxes = len(d['text'])
    #for i in range(n_boxes):
    #    if float(d['conf'][i]) > 60:
    #        x, y, w, h = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    #        text = d['text'][i]
    #        cv2.rectangle(frame, (x, y), (x + w , y + h), (0, 255, 255), 2)
    #        cv2.putText(frame, text, (x, y + h + 20), cv2.FONT_HERSHEY_DUPLEX,
    #                1, (0, 255, 255), 2)

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
            tempimage = frame[y * 2: y * 2 + h * 2, x * 2: x * 2 + w * 2]
            if bool_fast_decode: data = pdmtx.decode(tempimage, shrink=2, max_count=1)
            else: data = pdmtx.decode(tempimage, max_count=1)
            text = ""
            if len(data) > 0:
                text = data[0].data.decode('UTF-8')
                text = text.replace('\x1d', '\\x1d')
                print(f"found barcode: {text}")
                cv2.putText(frame, text, (x * 2, y * 2 + h * 2 + 20), cv2.FONT_HERSHEY_DUPLEX,
                    1, (0, 0, 255), 2)
                cv2.rectangle(frame, (x * 2, y * 2), (x * 2 + w * 2, y * 2 + h * 2), (0, 0, 255))
            
    cv2.imshow("image", frame)
    if bool_video and bool_save_video: images.append(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('I quit!')
        break

print('complite!')

if bool_video and bool_save_video:
    if len(images)>0:
        height, width, layers = images[1].shape
        video = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width, height))
        for i in range(len(images)):
            video.write(images[i])
        video.release()
    cap.release()

cv2.destroyAllWindows()
