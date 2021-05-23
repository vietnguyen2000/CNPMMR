import math
import cv2
from PIL import Image
import extcolors
from collections import Counter
import os
from shapedetector import ShapeDetector
from classify_icon import ClassifyIcon
from utils import reduceColor, isOverlap, mergeRect
import numpy as np

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
    classifyIcon = ClassifyIcon()
    acceptedContours = []
    for i in range(len(contours)):
        cnt = contours[i]
        rect = cv2.boundingRect(cnt)
        x, y, w, h = rect
        s = cv2.contourArea(cnt)
        condition = 0.0005 < w * h / area_img < 0.05 and \
                    0.1 < w / (h + 0.01) < 10 and \
                    0.78 < w * h / (s + 0.01) < 1.275
        if condition:
            isOverlapCheck = False
            for i in range(len(acceptedContours)):
                ac = acceptedContours[i]
                ac_ct = ac[0]
                ac_x, ac_y, ac_w, ac_h = ac[1]
                ac_s = ac[2]
                # check overlap
                if (isOverlap(rect, ac[1])):
                    isOverlapCheck = True

                    # remove the smaller one
                    if ac_s < s:
                        acceptedContours[i] = (cnt, (x, y, w, h), s)
                    break
            
            if not isOverlapCheck:
                acceptedContours.append((cnt, (x, y, w, h), s))
                    

    for ac in acceptedContours:
        cnt = ac[0]
        x, y, w, h = ac[1]
        s = ac[2]
        crop_img = img[y:y + h, x:x + w]
        crop_image = Image.fromarray(crop_img, "RGB")
        colors, pixel_count = extcolors.extract_from_image(crop_image)
        if len(colors) < 15:
            shape, radius, curvature = sd.detect(cnt)

            # putText(img, "{:0.2f}".format(h), (x, y + h /2), 2)
            # putText(img, "{:0.2f}".format(w), (x + w / 2, y + h), 1)
            # putText(img, "{:0.2f}".format(radius), (x + w, y + h), 0)

            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            

            reduce_crop_img = reduceColor(crop_img, 128)

            edge = cv2.Canny(reduce_crop_img, 0, 40)
            
            ret, thresh = cv2.threshold(edge, 0, 40, 0)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            icon_rect = None

            for icon_cnt in contours:
                rect = cv2.boundingRect(icon_cnt)
                icon_s = rect[2] * rect[3]
                # cv2.drawContours(img, icon_cnt + np.array([x,y]), -1, (255, 0, 0), 1)
                if icon_s / s > 0.05 and icon_s/s < 0.7:
                    if(icon_rect):
                        icon_rect = mergeRect(icon_rect, rect)
                    else:
                        icon_rect = rect

                    
                    

            if (icon_rect):
                icon_x, icon_y, icon_w, icon_h = icon_rect
                icon_x += x
                icon_y += y

                icon_s = icon_w * icon_h
                
                
                if (icon_s / s > 0.1 and icon_s / s < 0.7 and \
                    icon_w / icon_h <= 4 and icon_w / icon_h >= 1/4):
                    icon_img = img[icon_y : icon_y + icon_h, icon_x : icon_x + icon_w]
                    
                    # cv2.imshow('test', icon_img)
                    icon_type = classifyIcon.predict(icon_img, True)

                    cv2.drawContours(img, [cnt], -1, (0, 255, 0), 1)
                    putText(img, shape, (x + w/2, y - 5), 1)
                    putText(img, icon_type, (x + w/2, y + h + 15), 1)
                    # cv2.rectangle(img, (icon_x, icon_y), (icon_x + icon_w, icon_y + icon_h), (255,0,255),1)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    outPath = os.path.join(os.curdir, 'outputs', fileName)
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