import math
import cv2
from PIL import Image
import extcolors
from collections import Counter
import os
from shapedetector import ShapeDetector

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

    sd = ShapeDetector()

    acceptedContours = []
    for i in range(len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        s = cv2.contourArea(cnt)
        condition = 0.0001 < w * h / area_img < 0.05 and \
                    0.1 < w / (h + 0.01) < 10 and \
                    0.78 < w * h / (s + 0.01) < 1.275
        if condition:
            isOverlap = False
            for i in range(len(acceptedContours)):
                ac = acceptedContours[i]
                ac_ct = ac[0]
                ac_x, ac_y, ac_w, ac_h = ac[1]
                ac_s = ac[2]
                # check overlap
                if  max(x, ac_x) <= min(x + w, ac_x + ac_w) and \
                    max(y, ac_y) <= min(y + h, ac_y + ac_h):
                    isOverlap = True

                    # remove the smaller one
                    if ac_s < s:
                        acceptedContours[i] = (cnt, (x, y, w, h), s)
                    break
            
            if not isOverlap:
                acceptedContours.append((cnt, (x, y, w, h), s))
                    

    for ac in acceptedContours:
        cnt = ac[0]
        x, y, w, h = ac[1]
        s = ac[2]
        crop_img = img[y:y + h, x:x + w]
        crop_img = Image.fromarray(crop_img, "RGB")
        colors, pixel_count = extcolors.extract_from_image(crop_img)
        if len(colors) < 15:
            shape, radius, curvature = sd.detect(cnt)

            
            
            putText(img, shape, (x + w/2, y), 1)
            putText(img, "{:0.2f}".format(curvature), (x + w, y + h/2), 0)

            # putText(img, "{:0.2f}".format(h), (x, y + h /2), 2)
            # putText(img, "{:0.2f}".format(w), (x + w / 2, y + h), 1)
            # putText(img, "{:0.2f}".format(radius), (x + w, y + h), 0)

            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.drawContours(img, [cnt], -1, (0, 255, 0), 1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    outPath = os.path.join(os.curdir, 'buttons', fileName)
    cv2.imwrite(outPath, img)


TEXT_FACE = cv2.FONT_HERSHEY_DUPLEX
TEXT_SCALE = 0.5
TEXT_THICKNESS = 1
def putText(img, text, position, alignment = 0): 
    text_size, _ = cv2.getTextSize(text, TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
    if (alignment == 0): # align left
        text_origin = (int(position[0]), int(position[1]))
    elif(alignment == 1): # align middle
        text_origin = (int(position[0] - text_size[0] / 2), int(position[1]))
    else: # align right
        text_origin = (int(position[0] - text_size[0]), int(position[1]))
    img = cv2.putText(img, text, text_origin, TEXT_FACE, TEXT_SCALE, (0,0,255), TEXT_THICKNESS)