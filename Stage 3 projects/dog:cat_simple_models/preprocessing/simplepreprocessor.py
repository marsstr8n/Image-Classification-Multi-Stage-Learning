import cv2 as cv

# several methods to test out: resize ignoring aspect ratio, or take that into account
class SimplePreprocessor:
    def __init__(self, width, height, inter=cv.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter
        
    def preprocess(self, image):
        # resize the image to fixed width height, ignore aspect ratio
        return cv.resize(image, (self.width, self.height), interpolation=self.inter)