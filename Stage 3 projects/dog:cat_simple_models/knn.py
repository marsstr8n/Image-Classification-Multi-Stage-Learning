from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from datasets.simpledatasetloader import SimpleDatasetLoader
from preprocessing.simplepreprocessor import SimplePreprocessor
import glob
import os
import argparse
import numpy as np

# # construct the argument parse and parse the arguments - good for adding info in command line
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True,
# 	help="path to input dataset")
# ap.add_argument("-k", "--neighbors", type=int, default=1,
# 	help="# of nearest neighbors for classification")
# ap.add_argument("-j", "--jobs", type=int, default=-1,
# 	help="# of jobs for k-NN distance (-1 uses all available cores)")
# args = vars(ap.parse_args())

# hardcode it
args = {
    "dataset": "datasets/my_images",  # <-- your dataset folder
    "neighbors": 3,                   # <-- k value
    "jobs": -1                        # <-- use all CPU cores
}

# grab the list of images that we will be describing
print("[INFO] loading images...")

# get all image paths from both cats and dogs folders
imagePaths = glob.glob(os.path.join(args['dataset'], "*", "*")) 
    # dataset arg gotten from above
    # first * : matches the folder names: cats/ and dogs/
    # second * : matches the files inside those folders (.jpg, .png, etc)
    # glob.glob(...): finds all the matching pattern

# sort the paths to be consistent
imagePaths = sorted(imagePaths) 

sp = SimplePreprocessor(32, 32) # resize to 32x32 pixels
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072)) # 3072 = 32 x 32 x 3

# show some info on memory consumption of the images
print("[INFO] features matrix: {:.1f}MB".format(
    data.nbytes / (1024 * 1024.0)
))


# encode the label as intgers: change 'cats' and 'dogs' to 0 and 1 
le = LabelEncoder()
labels = le.fit_transform(labels)

# split the data 80/20
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)
    # use stratify to ensure equal proportion
    
# set up k value range
k_values = range(1,21)
best_k = None
best_accuracy = 0

print("[INFO] running cross validation to find the best k value")
for k_value in k_values:
    model = KNeighborsClassifier(n_neighbors=k_value, n_jobs=args["jobs"])
    # perform 5-fold CV - using accuracy as the metric of success
    cv_scores = cross_val_score(model, trainX, trainY, cv=5, scoring='accuracy')
    avg_accuracy = np.mean(cv_scores) 
    
    print(f"K={k_value}, Accuracy={avg_accuracy:.4f}")
    
    # Check if this k gives the best result
    if avg_accuracy > best_accuracy:
        best_accuracy = avg_accuracy
        best_k = k_value
    
# train and evaluate the result (no validation atm)
print(f"[INFO] evaluating KNN classifier... using best K value: {best_k}")
model = KNeighborsClassifier(n_neighbors=best_k, n_jobs=args["jobs"])
model.fit(trainX, trainY) # no actual learning - storing the trainX and trainY here - so it can predict
print(classification_report(testY, model.predict(testX), target_names=le.classes_))

# result evaluation: best K is 19 - accuracy score on test result is 0.62. Better result than no CV
