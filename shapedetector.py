import cv2
import math

class ShapeDetector:
    def __init__(self):
        pass
    def detect(self, c):
        shape = ''
        x, y, w, h = cv2.boundingRect(c)
        deltaS = w * h - cv2.contourArea(c)
        radius = math.sqrt(abs(deltaS / (4 - math.pi)))
        curvature = radius / min(w/2,h/2)

        # If bounding rect is like a square
        if (w * 0.95 < h < w * 1.05):
            if (curvature < 0.1):
                shape = 'Square'
            elif (curvature < 0.90):
                shape = 'Rounded Square'
            else:
                shape = 'Circle'
        else: 
            if (curvature < 0.1):
                shape = 'Rectangle'
            else:
                shape = 'Rounded Rectangle'
        return shape, radius, curvature
        