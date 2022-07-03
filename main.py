
import os
import cv2 as cv
from cv2 import dnn_DetectionModel
import numpy as np
import time


class Detector:
    def __init__(self, video_path, config_path, model_path, classes_path):
        self.video_path = video_path
        self.config_path = config_path
        self.model_path = model_path
        self.classes_path = classes_path

        self.net = cv.dnn_DetectionModel(self.model_path, self.config_path)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()

    def readClasses(self):
        with open(self.classes_path, 'r') as f:
            self.classes_list = f.read().splitlines()
        self.classes_list.insert(0, '__Background__')
        self.color_list = np.random.uniform(low=0, high=255, size=(
            len(self.classes_list), 3))
        print(self.classes_list)

    def onVideo(self):
        cap = cv.VideoCapture(self.video_path)
        if (cap.isOpened() == False):
            print('Fuck yourself, you cunt')
            return
        success, frame = cap.read()
        timeEnd = 0
        timeStart = 0

        while success:
            classLabelIds, confidences, bboxs = self.net.detect(
                frame, confThreshold=0.5)
            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1, -1)[0])
            confidences = list(map(float, confidences))
            bboxIdx = cv.dnn.NMSBoxes(
                bboxs, confidences, score_threshold=0.5, nms_threshold=0.2)
            if len(bboxIdx) != 0:
                for i in range(0, len(bboxIdx)):
                    bbox = bboxs[np.squeeze(bboxIdx[i])]
                    classConfidence = confidences[np.squeeze(bboxIdx[i])]
                    classLabelId = np.squeeze(
                        classLabelIds[np.squeeze(bboxIdx[i])])
                    classLabel = self.classes_list[classLabelId]
                    classColor = [int(c)
                                  for c in self.color_list[classLabelId]]
                    classColor = (0, 0, 255)

                    displayText = "{}:{:.4f}".format(
                        classLabel, classConfidence)
                    x, y, w, h = bbox

                    cv.rectangle(frame, (x, y), (x + w, y + h),
                                 color=classColor, thickness=1)
                    cv.putText(frame, displayText, (x, y - 10),
                               cv.FONT_HERSHEY_PLAIN, 1, classColor)

                    if classLabel == "cell phone":
                        cv.rectangle(frame, (x, y), (x + w, y + h),
                                     color=(0, 255, 0), thickness=1)
                        cv.putText(frame, displayText, (x, y - 10),
                                   cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

            cv.imshow("winname", frame)
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            success, frame = cap.read()

        cv.destroyAllWindows()


def main():
    video_path = 0
    config_path = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    model_path = "frozen_inference_graph.pb"
    classes_path = "coco.names"

    detector = Detector(video_path, config_path, model_path, classes_path)
    detector.onVideo()


if __name__ == '__main__':
    main()
