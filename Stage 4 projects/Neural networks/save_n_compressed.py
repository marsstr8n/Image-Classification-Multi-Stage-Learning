import numpy as np
from datasets.simpleimageloader import *
from preprocessing.simplepreprocessor import *
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import os
import glob

# set the args value:
args = {
    "dataset": "datasets/my_images",
    # something to do with logistic regression here
    "jobs": -1
}

# grab the list of imgs to be processed:
print("[INFO] loading images...")

# get all the imagePaths
imagePaths = glob.glob(os.path.join(args['dataset'], '*', '*'))

# sort the paths to be consistent
imagePaths = sorted(imagePaths)

sp = SimplePreprocessor(32,32)
sil = SimpleImageloader(preprocessors=[sp])
(imagedata, labels) = sil.load(imagePaths, verbose=500)
imagedata = imagedata.reshape(imagedata.shape[0], 3072)
print(imagedata[0])
# show some info on memory consumption of the images
print("[INFO] features matrix: {:.1f}MB".format(
    imagedata.nbytes / (1024 * 1024.0)
))

le = LabelEncoder()
labels = le.fit_transform(labels)
sc = StandardScaler()

# save the processed data for use later.
np.savez_compressed("preprocessed_data.npz", data=imagedata, labels=labels, class_names=le.classes_)


