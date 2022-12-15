import cv2
import time
from matplotlib import pyplot as plt


vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()
    plt.imshow(frame)
    time.sleep(5)
