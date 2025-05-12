import glob
import os
import cv2 as cv
import numpy as np

class SimpleImageloader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors
        
        if self.preprocessors is None:
            self.preprocessors = []
            
    def load(self, imagePaths, verbose=-1):
        """_summary_
        Args:
            imagePaths (list): specifies the file paths to the images in our dataset
            verbose (int, optional): verbosity level can be used to print updates to a console - monitor how many imgs has been processed. Defaults to -1.
        """
        imagedata = []
        labels = []
        
        # loop over the input images
        for (i, imagePath) in enumerate(imagePaths):
            # load the img and extract the class label - as in: dogs folder containing images of dogs, same with cats - separately
            # formatL /path/to/dataset/{class}/{image}.jpg
            image = cv.imread(imagePath) # load the image, and extract class label
            if image is None:
                print(f"[Warning] Unable to load image: {imagePath}")
                continue
            
            label = imagePath.split(os.path.sep)[-2] # extract the class label based on file path (-2 is {class})
            
            # preprocess
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image) # p in this case is SimplePreprocessor, and function SimplePreprocessor.preprocess(image)
                    
            # processed img is now a feature vector, append them to the list
            imagedata.append(image)
            labels.append(label)

            # show an update every 'verbose' images:
                # if 1000 images, verbose = 199 -> every 100 images, it will print a message
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))
                
        # return a tuple of the data and labels
        return (np.array(imagedata), np.array(labels))



