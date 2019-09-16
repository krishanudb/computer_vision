import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import cv2

from sklearn.cluster import MiniBatchKMeans

print('OpenCV 3.4.0.12 required. Install this, if current version is different')
print('Current OpenCV version: ',cv2.__version__)

folder = sys.argv[1]
# folder = '../input/insample'


files = os.listdir(folder)
pics = {}
kpics = {}
kp = {}
des = {}

nkp = {}
ndes = {}

PERCENTILE_CUTOFF = 80


for filee in files:
    image = cv2.imread(folder + "/" + filee)
    pics[filee.split(".")[0]] = image
    kpics[filee.split(".")[0]] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("All Images Read.")

for key, value in kpics.items():
    print("Extracting all features for Image {}".format(key))
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(kpics[key], None)
    kp[key] = keypoints
    des[key] = descriptors
    print("Finding unique features from Image {}".format(key))
    kmeans = MiniBatchKMeans(n_clusters=200, random_state=0, batch_size = len(kp[key]) / 100).fit(des[key])

    clusters = kmeans.labels_
    distances = np.sqrt(np.sum((des[key] - kmeans.cluster_centers_[clusters]) ** 2, axis = 1))

    percentile = np.percentile(distances, PERCENTILE_CUTOFF)
    ndes[key] = des[key][distances > percentile]
    nkp[key] = list(np.array(kp[key])[distances > percentile])
    
des = ndes
kp = nkp

print("Finding unique features done.")

print("Feature matching among all pairs of Images starting.")

matches_dict = {}
total_matches = []

for i in kpics.keys():
    matches_dict[i] = {}

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
            matches = good_matches        
            matches = sorted(matches, key = lambda x:x.distance)
            if len(matches) > 200: 
                matches = matches[:200]
            matches_dict[i][j] = matches
            total_matches += matches
            
            print("Between {} and {}, number of good matches: {}".format(i, j, len(matches)))
            
match_distances = [match.distance for match in total_matches]

percentile_matches = np.percentile(match_distances, 200. / len(files))

print("Feature Matching Done For all Images, Distance to beat: {}".format(percentile_matches))
            
good_matches_dict = {}
MIN_MATCH_COUNT = 100
for i in kpics.keys():
    good_matches_dict[i] = {} 
    for j in kpics.keys():
        if i != j:
            good_matches_dict[i][j] = []
            for match in matches_dict[i][j]:
                if match.distance < percentile_matches:
                    good_matches_dict[i][j].append(match)
             
            good_matches = good_matches_dict[i][j]
            
            if len(good_matches)>MIN_MATCH_COUNT:
                img1, img2, kp1, kp2 = kpics[i], kpics[j], kp[i], kp[j]
                cimg1, cimg2 = pics[i], pics[j]

                
                img1_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
                img2_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)                
                
                H1, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC,2)
                H2, mask = cv2.findHomography(img2_pts, img1_pts, cv2.RANSAC,2)
                if H1[0][2] > 0:
                    
                    H2_inverse = np.linalg.inv(H2)
                    
                    H = (H1 + H2_inverse) / 2.
                    
                    h1,w1 = img1.shape[:2]
                    h2,w2 = img2.shape[:2]
                    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
                    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)


                    pts2_ = cv2.perspectiveTransform(pts2, H)
                    pts = np.concatenate((pts1, pts2_), axis=0)

                    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
                    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)

                    t = [-xmin,-ymin]

                    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
                    
                    
                    result = cv2.warpPerspective(cimg1, Ht.dot(H), (xmax-xmin, ymax-ymin))
                    present_old = (result[t[1]:h1+t[1],t[0]:w1+t[0]] != 0).astype(np.int8)
                    result[t[1]:h1+t[1],t[0]:w1+t[0]] = (cimg2 + present_old * result[t[1]:h1+t[1],t[0]:w1+t[0]])/(1 + present_old)
                    
                    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    
                    # plt.figure(figsize=(20, 10))
                    # plt.imshow(result)
                    # plt.show()
                    
                    result = cv2.resize(result,None,fx=0.5,fy=0.5)

                    plt.imsave(folder + "/" +  "temp1_" + i + "_" + j + ".png", result)
                    print("Feature matching for images pairs {} and {} done".format(i, j))
            else:
                print ('Not enough matches are found between {} and {}'.format(i, j))
    print("Matching features for image {} with all image pairs Done".format(i))
print("Feature matching done for all images")
print("End !!")