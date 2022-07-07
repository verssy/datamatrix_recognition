
from time import sleep
import cv2
import pytesseract.pytesseract as pytess
import pylibdmtx.pylibdmtx as pdmtx
from colorama import Fore
import pyfiglet
import rich

pytess.tesseract_cmd=r'C:\Users\Дубровин\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# args
files_prefix = "./imgs/barcode_"
max_sides_ratio = 1.5
min_area = 600
delta_xy = 10
bool_save_video = False
bool_video = True
input_filename = 'vid_2.mp4'

#text_reader = easyocr.Reader(['en'], gpu=True)

files_count, images = 0, []
new_test_iters = [2, 2, 2, 2, 2, 2]
new_test_blurs = [1, 3, 1, 3, 1, 3]
new_test_erodes = [(3, 3), (3, 3), (5, 5), (5, 5), (7, 7), (7, 7)]
custom_config = r'-l deu --psm 6 tessedit_char_whitelist=0123456789/C'

if bool_video: cap = cv2.VideoCapture(f'./{input_filename}')
#cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap_aspect = 3

title = pyfiglet.figlet_format('Cold Factory', font='invita')
rich.print(f'[spring_green3]{title}[/spring_green3]')

while True:
    if bool_video:
        _, frame = cap.read()
        if not _: break
    else: frame = cv2.imread(f'{input_filename}')

## text recognition with EasyOCR
    #cv2.imwrite('temp.jpg', frame)
    #recognized = text_reader.readtext('temp.jpg', rotation_info=[180], decoder='beamsearch')
    #for (bbox, text, prob) in recognized:
    #    (tl, tr, br, bl) = bbox
    #    tl = (int(tl[0]), int(tl[1]))
    #    tr = (int(tr[0]), int(tr[1]))
    #    br = (int(br[0]), int(br[1]))
    #    bl = (int(bl[0]), int(bl[1]))
    #    if is_probably_date_string(text):
    #        cv2.rectangle(frame, tl, br, (0, 255, 0), 1)
    #        cv2.putText(frame, text, (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
    
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, None, fx=1/cap_aspect, fy=1/cap_aspect, interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
    harris = cv2.cornerHarris(img_gray, 4, 3, 0.06)
    __, thr = cv2.threshold(harris, 0.001 * harris.max(), 255, cv2.THRESH_BINARY)
    thr = thr.astype('uint8')

## pytesseract text recognition
    #assert len(new_test_iters) == len(new_test_blurs) == len(new_test_erodes)
    #len_cases = len(new_test_iters)
    #for i in range(len_cases):
    #    temp_frame = cv2.erode(frame, new_test_erodes[i], iterations=2)
    #    temp_frame = cv2.medianBlur(temp_frame, new_test_blurs[i])
    #    image_data = pytess.image_to_string(temp_frame, config=custom_config)
    #    print(image_data)
    #    #image_data = pytess.image_to_data(temp_frame, output_type=pytess.Output.DICT, config=custom_config)
    #    #for j in range(len(image_data['text'])):
    #    #    text = image_data['text'][j]
    #    #    if is_probably_date_string(text):
    #    #        x, y, w, h = (image_data['left'][j], image_data['top'][j], image_data['width'][j], image_data['height'][j])
    #    #        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    #    #        cv2.putText(frame, text, (x, y + h + 20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)

## pytesseract another variant
    #d = pytesseract.image_to_data(frame, output_type=pytesseract.Output.DICT)
    #n_boxes = len(d['text'])
    #for i in range(n_boxes):
    #    if float(d['conf'][i]) > 60:
    #        x, y, w, h = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    #        text = d['text'][i]
    #        cv2.rectangle(frame, (x, y), (x + w , y + h), (0, 255, 255), 2)
    #        cv2.putText(frame, text, (x, y + h + 20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)

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
            
    cv2.imshow("image", cv2.resize(frame, (0, 0), fx=0.5, fy=0.5))
    if bool_video and bool_save_video: images.append(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('I quit!')
        break

print('complite!')

if bool_video and bool_save_video:
    if len(images)>0:
        (height, width, layers) = images[1].shape
        video = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))
        for i in range(len(images)):
            video.write(images[i])
        video.release()
    cap.release()

cv2.destroyAllWindows()
