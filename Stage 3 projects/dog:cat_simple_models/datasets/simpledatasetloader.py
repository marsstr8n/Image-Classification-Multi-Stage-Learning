# goal: load images from the disks, preprocess each img. Returns images (raw pixels) and class labels for each image
# KNN, SVM, ... tend to require images to be in a fixed vector size i.e. identical width and heights

import numpy as np
import cv2 as cv
import os # extract the names of subdirectories in img paths

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None): # class SimplePreprocessor can be passed in here, or multiple of them
        # store the image preprocessor
        self.preprocessors = preprocessors

        # if no preprocessors, initialise as []
        if self.preprocessors is None:
            self.preprocessors = [] # why list, not as single value? could have several processes independently: resize, scaling, convertint to array,...
            
    def load(self, imagePaths, verbose=-1):
        """_summary_

        Args:
            imagePaths (list): specifies the file paths to the images in our dataset
            verbose (int, optional): verbosity level can be used to print updates to a console - monitor how many imgs has been processed. Defaults to -1.
        """
        # ini list of features and labels:
        data = []
        labels = []
        
        # loop over the input images
        for (i, imagePath) in enumerate(imagePaths):
            # load the img and extract the class label - as in: dogs folder containing images of dogs, same with cats - separately
            # formatL /path/to/dataset/{class}/{image}.jpg
            image = cv.imread(imagePath)
            if image is None:
                print(f"[Warning] Unable to load image: {imagePath}")
                continue
            label = imagePath.split(os.path.sep)[-2] # extract the class label based on file path (-2 is {class})
            
            # check to see preprocessors are not None:
            if self.preprocessors is not None:
                # loop over the preprocessors, apply to each of the image
                for p in self.preprocessors:
                    image = p.preprocess(image)
                    
            # processed img is now a feature vector, append them to the list
            data.append(image)
            labels.append(label)
            
            # show an update every 'verbose' images:
                # if 1000 images, verbose = 199 -> every 100 images, it will print a message
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))
                
        # return a tuple of the data and labels
        return (np.array(data), np.array(labels))
        
# encoe 
