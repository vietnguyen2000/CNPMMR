import cv2
import math
from PIL import Image
from skimage.morphology import medial_axis
from skimage import data
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from utils import reduceColor

class ClassifyIcon:
    
    def __init__(self):
        pass

    def predict(self, icon: Image, showDebug = False):
        # icon = cv2.cvtColor(icon, cv2.COLOR_BGR2RGB)
        h_0, w_0 = icon.shape[0], icon.shape[1]
        default = min(h_0, w_0)

        quantized = reduceColor(icon, 128)

        colors, counts = np.unique(quantized.reshape(-1,3), axis = 0, return_counts = True)
        colors_percent = counts / np.sum(counts)

        num_of_color = np.count_nonzero(colors_percent >= 0.1)
        
        if (num_of_color == 3):
            return 'Two tone'
        if (num_of_color <= 1 or num_of_color > 3):
            return 'Unknown'

        max_counts = max(counts)
        max_index = np.where(counts == max_counts)[0][0]
        max_colors = colors[max_index] # background

        data = np.apply_along_axis(lambda x: not self.compareColor(x, max_colors), -1, quantized)
        skel, distance = medial_axis(data, return_distance = True)
        
        thickness = np.max(distance) / default
        res = 'Out line' if thickness < 0.2 else 'Filled'
        if (showDebug): 
            print('Thickness:', thickness, '\nClass:', res)
            dist_on_skel = distance * skel
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
            ax1.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
            ax1.set_title('icon')

            ax2.imshow(dist_on_skel, cmap=plt.cm.nipy_spectral, interpolation='nearest')
            ax2.contour(data, [0.5], colors='w')
            ax2.set_title(res)
            # fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
            plt.show()

        return res

    def compareColor(self, compare, color):
        comparison = compare == color
        return comparison.all()
        
if __name__ == "__main__":
    classifyIcon = ClassifyIcon()
    filePath = os.path.join(os.curdir, sys.argv[1])
    dirname = os.path.basename(os.path.dirname(filePath))
    fileName = os.path.basename(filePath)
    icon = cv2.imread(filePath)

    out = classifyIcon.predict(icon, True)
# outPath = os.path.join(os.curdir, 'outputs', dirname, fileName)

# cv2.imwrite(outPath, out)