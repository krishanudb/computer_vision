# This the code is for feature extraction and matching. 
# This is a very time consuming process and I have tried to apply some heuristics to speed up the process.
# The heuristics are explained at the corresponding steps

# Input: python fem.py **folder_name_containing_all_pictures**

import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import cv2


folder = sys.argv[1]

files = os.listdir(folder)
pics = {}
kpics = {}

kp = {}
des = {}

for filee in files:
    pics[filee.split(".")[0]] = cv2.cvtColor(cv2.imread(folder + "/" + filee), cv2.COLOR_BGR2RGB)
    kpics[filee.split(".")[0]] = cv2.cvtColor(cv2.imread(folder + "/" + filee), cv2.COLOR_BGR2GRAY)
print("All Images Read.")

# FIRST HEURISTIC:
# Since the number of features extracted by SIFT is very high, we need to remove the repetitive features.
# Having too many repetitive/similar features in one image increases the time for feature matching and also gives inaccurate matches.
# The way we tried to reduce repetitive features is as follows:
# We ran KMeans clustering on the features.
# The feature points which are far away from their corresponding cluster centroids can be thought of as outliers.
# The outliers can be assumed to be fairly different from the other features.
# We considered only the outlier features for our algo.

# Step 1: We clustered all the features obtained from a single image into a very high number of clusters (200).
# Step 2: We computed the distance of every feature from the corresponding cluster centroid
# Step 3: We found a distribution of the distances from centroid and converted into a probability density function.
# Step 4: We found the 90th percentile distance, and selected only the features whose distances from their corresponding centroids
# is more than the 90th percentile distance.
# Step 5: These selected features are the outliers, and only they were considered for further processing.


from sklearn.cluster import MiniBatchKMeans

nkp = {}
ndes = {}

for key, value in kpics.items():
    print("Extracting features for Image {}".format(key))
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(kpics[key], None)
    kp[key] = keypoints
    des[key] = descriptors
    print("Running KMeans on features from Image {}".format(key))
    kmeans = MiniBatchKMeans(n_clusters=200, random_state=0, batch_size = len(kp[key]) / 100).fit(des[key])

    clusters = kmeans.labels_
    distances = np.sqrt(np.sum((des[key] - kmeans.cluster_centers_[clusters]) ** 2, axis = 1))

    percentile_90 = np.percentile(distances, 90)
    # percentile_99 = np.percentile(distances, 99)
    # percentile_999 = np.percentile(distances, 99.9)


    ndes[key] = des[key][distances > percentile_90]
    nkp[key] = list(np.array(kp[key])[distances > percentile_90])
    
    # kpics[key]= cv2.drawKeypoints(kpics[key], nkp[key], None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

print("Clustering and finding unique features done.")

# SECOND HEURISTIC
# The feature matching step produces too many matches, most of them being spurious.
# Additionally, we do not know beforehand the ordering of the images, so we have to find the best matching images first
# We do this by finding the images with best matching features in the following way:
# Step 1: We find the best 200 feature matches for every pair of images.
# Step 2: We find the distances of each match (euclidean distance between the feature in one image and the most similar feature in the other image)
# Step 3: We combine all the distances from all image pairs and find the 20th percentile value.
# Step 4: We select only those feature matches whose distance is less than the 20th percentile value.
# These features are the best matching features across all pairs of images and their number (between every image pair) gives a 
# measure of the match between the two images
print("Feature matching among all pairs of Images starting.")
total_matches = []
matches_dict = {}
for i in kpics.keys():
    matches_dict[i] = {}
    for j in kpics.keys():
        if i != j:
            key1 = i
            key2 = j
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(ndes[key1] ,ndes[key2])

            matches = sorted(matches, key = lambda x:x.distance)
            matches = matches[:200]
            matches_dict[i][j] = matches
            total_matches += matches
    print("Matching features for image {} Done".format(i))

match_distances = [match.distance for match in total_matches]
percentile_matches_20 = np.percentile(match_distances, 20)

print("Feature Matching Done For all Images, Distance to beat: {}".format(percentile_matches_20))



good_matches = {}

for i in kpics.keys():
    good_matches[i] = {} 
    for j in kpics.keys():
        if i != j:
            good_matches[i][j] = []
            for match in matches_dict[i][j]:
                if match.distance < percentile_matches_20:
                    good_matches[i][j].append(match)
            print("Between {} and {}, number of good matches: {}".format(i, j, len(good_matches[i][j])))
            img3 = cv2.drawMatches(pics[i],nkp[i],pics[j],nkp[j],good_matches[i][j], None, flags=2)
            plt.imsave(folder + "/" +  "temp_" + i + "_" + j + ".png", img3)
            print("Feature match Image for images {} and {} saved as {}".format(i, j, folder + "/" +  "temp_" + i + "_" + j + ".png"))

