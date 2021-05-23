import cv2
import math
from PIL import Image
from skimage.morphology import medial_axis
from skimage import data
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

class ClassifyIcon:
    
    def __init__(self):
        pass

    def predict(self, icon: Image, showDebug = False):
        # icon = cv2.cvtColor(icon, cv2.COLOR_BGR2RGB)
        h_0, w_0 = icon.shape[0], icon.shape[1]
        default = min(h_0, w_0)

        quantized = self.reduceColor(icon)

        colors, counts = np.unique(quantized.reshape(-1,3), axis = 0, return_counts = True)
        
        if (len(colors) >= 3):
            return 'Two tone'

        max_counts = max(counts)
        max_index = np.where(counts == max_counts)[0][0]
        max_colors = colors[max_index] # background

        data = np.apply_along_axis(lambda x: not self.compareColor(x, max_colors), -1, quantized)
        skel, distance = medial_axis(data, return_distance = True)
        
        thickness = np.max(distance) / default

        if (showDebug): 
            print('thick ness:', thickness)
            dist_on_skel = distance * skel
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
            ax1.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
            ax1.axis('off')
            ax2.imshow(dist_on_skel, cmap=plt.cm.nipy_spectral, interpolation='nearest')
            ax2.contour(data, [0.5], colors='w')
            ax2.axis('off')
            fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
            plt.show()

        return 'Out line' if thickness < 0.1 else 'filled'

    def reduceColor(self, icon: Image, div = 64):
        return icon // div * div + div // 2

    def compareColor(self, compare, color):
        comparison = compare == color
        return comparison.all()
        

classifyIcon = ClassifyIcon()
filePath = os.path.join(os.curdir, sys.argv[1])
dirname = os.path.basename(os.path.dirname(filePath))
fileName = os.path.basename(filePath)
icon = cv2.imread(filePath)

out = classifyIcon.predict(icon, True)
# outPath = os.path.join(os.curdir, 'outputs', dirname, fileName)

# cv2.imwrite(outPath, out)