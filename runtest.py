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

def detect_button(filePath, json):
    fileName = os.path.basename(filePath)
    # Read image and resize
    img_path = filePath  # Input the path of the image
    img = cv2.imread(img_path)
    # img = reduceColor(img, 128)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # ratio = 1920 / img.shape[1]
    # img = cv2.resize(img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)))  # Resize the image to (1920,)
    h_0, w_0 = img.shape[0], img.shape[1]  # height, width of the image after resize
    area_img = h_0 * w_0  # area (number of pixels) of the image

    # img = img[int(json['x1']) : int(json['x2'])][int(json['y1']) : int(json['y2'])]
    # initial parameter setting
    threshold1 = 0
    threshold2 = 40
    img_edge = cv2.Canny(img, threshold1, threshold2)
    # cv2.imshow('img_gray', img_edge)
    # cv2.waitKey(0)
    # area = 1200

    # find Contours
    ret, thresh = cv2.threshold(img_edge, threshold1, threshold2, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
    overall_radius = []

    sd = ShapeDetector()
    classifyIcon = ClassifyIcon()
    for button in json:
        acceptedContours = []
        for i in range(len(contours)):
            cnt = contours[i]
            
            rect = cv2.boundingRect(cnt)
            x, y, w, h = rect
            if (x <= button['x1'] or y <= button['y1'] or x + w >= button['x2'] or y + h >= button['y2']): continue
            # cv2.drawContours(img, [cnt], -1, (0, 255, 0), 1)
            s = cv2.contourArea(cnt)
            condition = w > 0 and h > 0 and s > 0
            
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

            shape, radius, curvature = sd.detect(cnt,)
            
            icon = button['icon']
            icon_rect = (icon['x1'], icon['y1'], icon['x2'] - icon['x1'], icon['y2'] - icon['y1'])

            icon_x, icon_y, icon_w, icon_h = icon_rect

            icon_s = icon_w * icon_h
            
            icon_img = img[icon_y : icon_y + icon_h, icon_x : icon_x + icon_w]
            # print(img.shape)
            
            # cv2.imshow('test', icon_img)
            icon_type = classifyIcon.predict(icon_img)

            cv2.drawContours(img, [cnt], -1, (0, 255, 0), 1)
            putText(img, shape, (x + w/2, y - 5), 1)
            putText(img, icon_type, (x + w/2, y + h + 15), 1)
            cv2.rectangle(img, (icon_x, icon_y), (icon_x + icon_w, icon_y + icon_h), (255,0,255),1)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    dirPath = os.path.join(os.curdir, 'outputs')
    outPath = os.path.join(dirPath, fileName)
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
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


import sys 
import json

def main():
    if (len(sys.argv) < 2): return
    path = os.path.join(os.curdir, sys.argv[1])
    if (os.path.isfile(path)):
        fileName = os.path.splitext(path)[0]
        data = json.load(open( fileName + '.json'))
        detect_button(path, data)
    
    elif (os.path.isdir(path)):
        for entry in os.scandir(path):
            if (entry.path.endswith('jpg') or entry.path.endswith('png')):
                fileName = os.path.splitext(entry.path)[0]
                data = json.load(open( fileName + '.json'))
                detect_button(entry.path, data)



if __name__ == "__main__":
    main()  