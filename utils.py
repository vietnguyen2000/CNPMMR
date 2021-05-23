from PIL import Image
def reduceColor(icon: Image, div = 64):
    return icon // div * div + div // 2

def isOverlap(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    if  max(x1, x2) <= min(x1 + w1, x2 + w2) and \
        max(y1, y2) <= min(y1 + h1, y2 + h2):
        return True
    return False

def mergeRect(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    newX = min(x1, x2)
    newY = min(y1, y2)
    newWidth = max(x1 + w1, x2 + w2) - newX
    newHeight = max(y1 + h1, y2 + h2) - newY
    return (newX, newY, newWidth, newHeight)