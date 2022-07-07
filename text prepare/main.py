
import time
import cv2
import numpy as np
import pytesseract.pytesseract as tess

tess.tesseract_cmd=r'C:\Users\Дубровин\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def remove_noise(image):
    return cv2.medianBlur(image,5)
 
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)

def erode(image):
    kernel = np.ones((3,3),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def canny(image):
    return cv2.Canny(image, 100, 200)

def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

arr_blurs = [1, 3, 5]
arr_erodes = [(3, 3), (5, 5), (7, 7), (9, 9)]

test_iters = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]
test_blurs = [5, 5, 5, 5, 1, 3, 5, 1, 3, 5, 1, 3, 5]
test_erodes = [(3, 3), (5, 5), (7, 7), (9, 9), (3, 3), (3, 3), (3, 3), (5, 5), (5, 5), (5, 5), (7, 7), (7, 7), (7, 7)]

new_test_iters = [2, 2, 2, 2, 2, 2]
new_test_blurs = [1, 3, 1, 3, 1, 3]
new_test_erodes = [(3, 3), (3, 3), (5, 5), (5, 5), (7, 7), (7, 7)]

def main():
    custom_config = r'-l deu --psm 6 tessedit_char_whitelist=0123456789/C'

    is_training = False
    frame = cv2.imread('test4.jpg')
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    if is_training:
## TRAINING
        len_blurs = len(arr_blurs)
        len_erodes = len(arr_erodes)
        iteration = 0
        for i in range(len_blurs * len_erodes * 2):
            temp_frame = frame.copy()
            eroded = cv2.erode(temp_frame, arr_erodes[int(i % (len_blurs * len_erodes) / len_blurs)], iterations=1 + int(i / (len_blurs * len_erodes)))
            temp_frame = cv2.medianBlur(eroded, arr_blurs[i % (len_blurs * len_erodes) % len_blurs])
            image_data = tess.image_to_string(temp_frame, config=custom_config)
            iteration = iteration + 1
            print(f'{iteration}: erode_iters={i / (len_blurs * len_erodes)}, blur={arr_blurs[i % (len_blurs * len_erodes) % len_blurs]}, erode={arr_erodes[int(i % (len_blurs * len_erodes) / len_blurs)]}, data={image_data}')
    else:
## TESTING
        assert len(new_test_iters) == len(new_test_blurs) == len(new_test_erodes)
        len_cases = len(new_test_iters)
        time_start = time.time()
        for i in range(len_cases):
            temp_frame = cv2.erode(frame, new_test_erodes[i], iterations=new_test_iters[i])
            temp_frame = cv2.medianBlur(temp_frame, new_test_blurs[i])
            image_data = tess.image_to_string(temp_frame, config=custom_config)
            print(f'image_data: {image_data}')
        time_end = time.time()
        print(f'time: {time_end - time_start}')

    #frame = get_grayscale(frame)
    #frame = thresholding(frame)
    #frame = erode(frame)
    #frame = remove_noise(frame)
    
    #print(tess.image_to_string(frame))

    cv2.destroyAllWindows()


if __name__=='__main__':
    main()