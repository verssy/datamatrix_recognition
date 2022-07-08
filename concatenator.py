
import cv2

vids = ['demo.avi', 'output.avi']
images = []
for i in range(60):
    images.append(cv2.imread('cat_1.jpg'))
for i in range(60):
    images.append(cv2.imread('cat_2.jpg'))
for i in range(60):
    images.append(cv2.imread('cat_3.jpg'))

(height, width, layers) = images[1].shape
video = cv2.VideoWriter('ColdFucktory.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))

for vid in vids:
    cap = cv2.VideoCapture(vid)
    while True:
        _, frame = cap.read()
        if not _: break
        frame = cv2.resize(frame, (width, height))
        video.write(frame)
    cap.release()

