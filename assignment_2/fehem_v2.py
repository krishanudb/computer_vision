# This code is for feature extraction, homography estimation and matching. 
# The strategy used is described in steps below:
# Step 1: Extract the features using SIFT and find unique features using the method previously described.
# Step 2: Match the features using FLANN (this is fastest for large datasets)
# Step 3: Store the best matches using Lowe's ratio test (gievn in SIFT Paper), which will then be considered for homography estimation
# (Note: The Image pairs with best matches greater than 300 will be considered for panorma construction 
# 300 is decided based upon my own matching analysis. As per assignment, we need to come up with some scoring mechanism for decinding this threshold
# Can we mention the clustering done in previous version of code as a scoring mechanism ?)
# Step 4: Compute a homography via RANSAC
# Step 5: Used the homography to wrap the pair of images using perspective transform.

# Input: python fehem.py **folder_name_containing_all_pictures**

import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import cv2

from sklearn.cluster import MiniBatchKMeans

print('OpenCV 3.4.0.12 required. Install this, if current version is different')
print('Current OpenCV version: ',cv2.__version__)

# folder = sys.argv[1]
folder = '../input/insample'

files = os.listdir(folder)
pics = {}
kpics = {}
kp = {}
des = {}

nkp = {}
ndes = {}

for filee in files:
    pics[filee.split(".")[0]] = cv2.imread(folder + "/" + filee,0)
    kpics[filee.split(".")[0]] = cv2.imread(folder + "/" + filee,0)
print("All Images Read.")

for key, value in kpics.items():
    print("Extracting all features for Image {}".format(key))
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(kpics[key], None)
    kp[key] = keypoints
    des[key] = descriptors
    print("Finding unique features from Image {}".format(key))
    kmeans = MiniBatchKMeans(n_clusters=200, random_state=0, batch_size = len(kp[key]) // 100).fit(des[key])

    clusters = kmeans.labels_
    distances = np.sqrt(np.sum((des[key] - kmeans.cluster_centers_[clusters]) ** 2, axis = 1))

    percentile_90 = np.percentile(distances, 90)

    ndes[key] = des[key][distances > percentile_90]
    nkp[key] = list(np.array(kp[key])[distances > percentile_90])
    
des = ndes
kp = nkp

print("Finding unique features done.")

print("Feature matching among all pairs of Images starting.")
for i in kpics.keys():
    for j in kpics.keys():
        if i != j:
            key1 = i
            key2 = j
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)   # or pass empty dictionary
            flann = cv2.FlannBasedMatcher(index_params,search_params)
            matches = flann.knnMatch(des[key1],des[key2],k=2)
            matchesMask = [[0,0] for _ in range(len(matches))]
            good_matches = []
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good_matches.append(m)
            print("Between {} and {}, number of good matches: {}".format(i, j, len(good_matches)))
            
            MIN_MATCH_COUNT=300  
            if len(good_matches)>MIN_MATCH_COUNT:
                img1, img2, kp1, kp2=pics[i], pics[j], kp[i], kp[j]
                img1_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
                img2_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

                H, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC,5.0)
                matchesMask = mask.ravel().tolist()

                h,w = img1.shape
                crnr_pts_img1 = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                crnr_pts_img2 = cv2.perspectiveTransform(crnr_pts_img1,H)
#                 img2 = cv2.polylines(img2,[np.int32(crnr_pts_img2)],True,(0,255,0),10, cv2.LINE_AA)
#                 draw_params = dict(matchColor = (0,255,0),singlePointColor = None,matchesMask = matchesMask, flags = 2)
#                 result_img = cv2.drawMatches(img1,kp1,img2,kp2,good_matches,None,**draw_params)
                result_img = cv2.warpPerspective(img1,M,(img2.shape[1] + img1.shape[1], img2.shape[0]))
                result_img[0:img2.shape[0], 0:img2.shape[1]] = img2
                plt.imshow(result_img),plt.show()
#                 plt.imsave(folder + "/" +  "temp1_" + i + "_" + j + ".png", result_img)
                print("Feature matching for images pairs {} and {} done".format(i, j))
            else:
                print ('Not enough matches are found between {} and {}'.format(i, j))
    print("Matching features for image {} with all image pairs Done".format(i))
print("Feature matching done for all images")
print("End !!")
