import cv2 as cv
import numpy as np

class SimplePreprocessor:
    def __init__(self, width, height, inter=cv.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter
      
    # this method will resize the image while keeping the aspect ratio - the rest will be padding    
    def preprocess(self, image):
        h, w = image.shape[:2]
        scale = min(self.height/h, self.width/w) # the scale shall take the min value of the 2 ratios
        new_h = int(scale * h)
        new_w = int(scale * w)
        
        resized = cv.resize(image, (new_w, new_h), interpolation = self.inter)
        
        # create a blank canvas, where the resized image will sit on in the middle - pad the remaining space as black
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # place the image in the middle - IF image dim is smaller than the specified size - we keep track of how much
        # should the image be offset on both sides - ensuring that the image is placed in the middle of canvas
        x_offset = (self.width - new_w) // 2 # example: (32-28) // 2 = 2
        y_offset = (self.height - new_h) // 2 # (32-32) // 2 = 0
        
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        # canvas [0:32, 2:30] = resized. So the image has paddings on the x-axis, each side being 2.
        
        return canvas
        
        