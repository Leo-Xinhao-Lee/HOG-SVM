import numpy as np
import imutils
import cv2

def pyramid_hand(img, windowSize, downscale=1.5):
    scale_times = 0
    yield img

    while img.shape[0] >= windowSize[1] and img.shape[1] >= windowSize[0]:
        w = int(img.shape[1] / downscale)
        img = imutils.resize(img, width=w)
        yield img
