import math
import cv2
from PIL import Image
import extcolors
from collections import Counter
import os

def detect_button(filePath):
    fileName = os.path.basename(filePath)
    # Read image and resize
    img_path = filePath  # Input the path of the image
    img = cv2.imread(img_path)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ratio = 1920 / img.shape[1]
    img = cv2.resize(img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)))  # Resize the image to (1920,)
    h_0, w_0 = img.shape[0], img.shape[1]  # height, width of the image after resize
    area_img = h_0 * w_0  # area (number of pixels) of the image


    # initial parameter setting
    threshold1 = 0
    threshold2 = 40
    img_edge = cv2.Canny(img, threshold1, threshold2)
    # cv2.imshow('img_gray', img_edge)
    # cv2.waitKey(0)
    area = 1200

    # find Contours
    ret, thresh = cv2.threshold(img_edge, threshold1, threshold2, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    overall_radius = []
    for i in range(len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        condition = 0.001 < w * h / area_img < 0.05 and 0.1 < w / (h + 1) < 3 and 0.78 < w * h / (
                cv2.contourArea(cnt) + 1) < 1.275 and 0.02 < h / h_0 < 0.4
        if condition:
            crop_img = img[y:y + h, x:x + w]
            crop_img = Image.fromarray(crop_img, "RGB")
            colors, pixel_count = extcolors.extract_from_image(crop_img)
            if len(colors) < 15:
                deltaS = (w - 1) * (h - 1) - cv2.contourArea(cnt)
                radius = math.sqrt(abs(deltaS / (4 - math.pi)))
                curvature = round(radius / h, 2)
                # if curvature > 0.4:
                #     overall_radius.append('Roundest')
                # elif curvature > 0.25:
                #     overall_radius.append('Rounder')
                # elif curvature > 0.1:
                #     overall_radius.append('Round')
                # else:
                #     overall_radius.append('Normal')
                img = cv2.putText(img, str(curvature), (x + w, y + int(h / 2)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.drawContours(img, [cnt], -1, (0, 255, 0), 1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    outPath = os.path.join(os.curdir, 'buttons', fileName)
    cv2.imwrite(outPath, img)