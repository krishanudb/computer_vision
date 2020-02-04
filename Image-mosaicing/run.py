import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import cv2
import math

folder = sys.argv[1]

if folder[-1] == "/":
    folder = folder[:-1]
panoname = folder.split("/")[-1]

files = os.listdir(folder)
pics = {}
kpics = {}

PERCENTILE_CUTOFF = 50
RANSAC_CUTOFF = 10
RANSAC_ITERATIONS = 2000

for filee in files:
    image = cv2.imread(folder + "/" + filee)
    image = cv2.resize(image,None,fx=0.10,fy=0.10)

    lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    image = final

    pics[filee.split(".")[0]] = image
    kpics[filee.split(".")[0]] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("All Images Read.")

kp = {}
des = {}

nkp = {}
ndes = {}

for key, value in kpics.items():
    print("Extracting all features for Image {}".format(key))
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(kpics[key], None)
    kp[key] = keypoints
    des[key] = descriptors

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
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            matches = good_matches        
            matches = sorted(matches, key = lambda x:x.distance)
            
            if len(matches) > 200:
                matches = matches[:200]
            matches_dict[i][j] = matches
            total_matches += matches
            
            # print("Between {} and {}, number of good matches: {}".format(i, j, len(matches)))

average_matches_per_pair = len(total_matches) / (len(matches_dict) **2  - len(matches_dict))

match_distances = [match.distance for match in total_matches]

# print(100 / (len(kpics) - 1))

percentile_matches = np.percentile(match_distances, 100 / (len(kpics) - 1))

# print("Feature Matching Done For all Images, Distance to beat: {}".format(percentile_matches))

MIN_MATCH_COUNT = average_matches_per_pair / (2 * len(kpics) - 2)

# print(MIN_MATCH_COUNT)
best_match = {}
best_match_count = {}
good_matches_dict = {}

for key in kpics.keys():
    best_match_count[key] = 0

order = {}
for i in kpics.keys():
    good_matches_dict[i] = {} 
    for j in kpics.keys():
        if i != j:
            good_matches_dict[i][j] = []
            for match in matches_dict[i][j]:
                if match.distance < percentile_matches:
                    good_matches_dict[i][j].append(match)
             
            good_matches = good_matches_dict[i][j]
            # print("Between {} and {}".format(i, j))
            # print(len(good_matches))
            
            if len(good_matches) >= MIN_MATCH_COUNT:
                img1, img2, kp1, kp2 = kpics[i], kpics[j], kp[i], kp[j]
                cimg1, cimg2 = pics[i], pics[j]
                
                img1_pts = np.float32([kp1[m.queryIdx].pt for m in matches_dict[i][j]]).reshape(-1,1,2)
                img2_pts = np.float32([kp2[m.trainIdx].pt for m in matches_dict[i][j]]).reshape(-1,1,2)                
                H1, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC,RANSAC_CUTOFF, maxIters=RANSAC_ITERATIONS)

                if type(H1) != type(None) and H1[0][2] > 0 and H1[1][2] > 0:
                    H2, mask = cv2.findHomography(img2_pts, img1_pts, cv2.RANSAC,RANSAC_CUTOFF, maxIters=RANSAC_ITERATIONS)

                    H2_inverse = np.linalg.inv(H2)

                    H = (H1 + H2_inverse) / 2.
                    
                    
                    if len(good_matches) > best_match_count[j]:
                        best_match[j] = (i, H)
                        best_match_count[j] = len(good_matches)
                        order[j] = i


present_key = []
present_target = []
new_best_match = {}
for key, value in best_match.items():
    target = value[0]
    # print(key, target)
    if target not in present_target and key not in present_key:
        new_best_match[key] = value
        present_key.append(key)
        present_target.append(target)
        
best_match = new_best_match

def blend_images(image1, image2):
    value1 = (image1.sum(axis = -1) != 0).astype(np.int8)
    value2 = (image2.sum(axis = -1) != 0).astype(np.int8)
    
    overlap = ((value1 + value2) == 2).astype(np.int8)
    # print(overlap.sum())
    
    if overlap.sum() > 0:
        xmin = np.nonzero(overlap)[1].min() - 2
        xmax = np.nonzero(overlap)[1].max() + 2

        x_overlap = np.zeros(overlap.shape)
        for i in range(0, x_overlap.shape[1]):
            x_overlap[:, i] = i

        no = overlap * (x_overlap - xmin) / (xmax - xmin)
        on = (1 - no) * overlap

        mask1 = no + np.array(no == 0, dtype=np.int8)

        new_img1 = image1 * mask1.reshape(mask1.shape[0], mask1.shape[1], 1)
        del image1, mask1
        new_img1 = new_img1.astype(np.uint8)

        mask2 = on + np.array(on == 0, dtype=np.int8)

        new_img2 = image2 * mask2.reshape(mask2.shape[0], mask2.shape[1], 1)
        del image2, mask2
        new_img2 = new_img2.astype(np.uint8)

        result_new = new_img1 + new_img2

        # plt.imshow(result_new)
        # plt.show()
        return result_new

    else:
        result_new = image1 + image2
        
        # plt.imshow(result_new)
        # plt.show()
        return result_new


def size_middle(middle, best_match, order_list):
#     print(order_list)
    order_list_temp = order_list[:]
    print("For Middle as {}".format(order_list[middle]))
    H_dict = {}
    H_dict[order_list[middle]] = np.eye(3)

    for i in range(middle):
        _, H_i = best_match[order_list_temp[i]]
        for j in range(i + 1, middle):
            _, H_j = best_match[order_list_temp[j]]

            H_i = np.dot(H_i, H_j)

        H_i = np.linalg.inv(H_i)
        H_dict[order_list_temp[i]] = H_i


    for i in order_list[middle:]:
        if i in best_match.keys():
            j, H = best_match[i]
#             print(H_dict)
            H_dict[j] = np.dot(H_dict[i], H)

    best_match_new = {}
    for key in matches_dict.keys():
        if key not in order_list:
#             print(key)
            max_matches = 0
            for match in matches_dict[key].keys():
                if match in order_list:

                    if len(matches_dict[key][match]) > max_matches:
                        best_match_new[key] = match
                        max_matches = len(matches_dict[key][match])



            i = best_match_new[key]
            j = key

            img1, img2, kp1, kp2 = pics[i], pics[j], kp[i], kp[j]

            img1_pts = np.float32([ kp1[m.queryIdx].pt for m in matches_dict[i][j]]).reshape(-1,1,2)
            img2_pts = np.float32([ kp2[m.trainIdx].pt for m in matches_dict[i][j]]).reshape(-1,1,2)                

            H, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC,RANSAC_CUTOFF, maxIters=RANSAC_ITERATIONS)

            H = np.linalg.inv(H)
            H1 = np.dot(H_dict[best_match_new[key]], H)
            H_dict[key] = H1
            order_list_temp.append(key)
#             print(order_list)


    total_corner_pts = []
    for i in order_list_temp:
        img = pics[i]
        h,w = img.shape[:2]
        pts = np.float32([[0,0],[0,h],[w,0],[w,h]]).reshape(-1, 1, 2)
        pts_t = cv2.perspectiveTransform(pts, H_dict[i])
        angle =  np.absolute(math.atan2(H_dict[i][0,1], H_dict[i][0,0]) * 180/ np.pi)

        # print(pts_t.reshape((-1, 2)))
        # print("\n")
        xs = pts_t.reshape((-1, 2))[:, 0]
        ys = pts_t.reshape((-1, 2))[:, 1]
        if xs[0] > xs[2] or xs[1] > xs[3]:
            print("Image Flipping")
            return np.inf
        if ys[0] > ys[1] or ys[2] > ys[3]:
            print("Image Flipping")
            return np.inf
        # # print(xs)
        if angle > 90:
            print("Image Flipping")
            return np.inf
#         print(angle)
        total_corner_pts.append(pts_t)


    pts = np.concatenate(total_corner_pts, axis=0)


    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
#     print(xmin, ymin, xmax, ymax)

    dsize = (xmax - xmin, ymax - ymin)

    size = dsize[0] + dsize[1]
#     print(size)
    return size


for key in order.keys():
    if key not in order.values():
        starting_frame = key
        
reverse_order = {}
for key in order.keys():
    value = order[key]
    reverse_order[value] = key
    
key = starting_frame

num = 0
order_list = [key]


while key in best_match.keys():
    match, H = best_match[key]
    key = match
    order_list.append(key)
    
# middle0 = int(len(order_list) / 2)
# middle1 = int(len(order_list) / 2 - 1)

print("Finding the middle Image")

# if size_middle(middle0, best_match, order_list) > size_middle(middle1, best_match, order_list):
#     middle = middle1
# else:
#     middle = middle0
middle = None
min_size = np.inf
for i in range(len(order_list)):
    size = size_middle(i, best_match, order_list)
    if size < min_size:
        min_size = size
        middle = i
if type(middle) == type(None):
    print("\nCamera Pan too High. Result wont be Reliable.\n")
    middle = int(len(order_list) / 2 - 1)

print("Middle Image {}".format(order_list[middle]))
H_dict = {}
H_dict[order_list[middle]] = np.eye(3)

for i in range(middle):
    _, H_i = best_match[order_list[i]]
    for j in range(i + 1, middle):
        _, H_j = best_match[order_list[j]]

        H_i = np.dot(H_i, H_j)

    H_i = np.linalg.inv(H_i)
    H_dict[order_list[i]] = H_i

    
for i in order_list[middle:]:
    if i in best_match.keys():
        j, H = best_match[i]
        H_dict[j] = np.dot(H_dict[i], H)
    
for key in matches_dict.keys():
    if key not in order_list:
#         print(key)
        max_matches = 0
        for match in matches_dict[key].keys():
            if match in order_list:

                if len(matches_dict[key][match]) > max_matches:
                    best_match[key] = match
                    max_matches = len(matches_dict[key][match])

                
        
        i = best_match[key]
        j = key
        
        img1, img2, kp1, kp2 = pics[i], pics[j], kp[i], kp[j]

        img1_pts = np.float32([ kp1[m.queryIdx].pt for m in matches_dict[i][j]]).reshape(-1,1,2)
        img2_pts = np.float32([ kp2[m.trainIdx].pt for m in matches_dict[i][j]]).reshape(-1,1,2)                
        
        H, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC,RANSAC_CUTOFF, maxIters=RANSAC_ITERATIONS)

        H = np.linalg.inv(H)
        H1 = np.dot(H_dict[best_match[key]], H)
        H_dict[key] = H1
        order_list.append(key)
#         print(order_list)
        
total_corner_pts = []

print("Finding Panorama Dimensions")
for i in order_list:
    img = pics[i]
    # img = cv2.resize(img,None,fx=0.10,fy=0.10)
    h,w = img.shape[:2]
    pts = np.float32([[0,0],[0,h],[w,0],[w,h]]).reshape(-1, 1, 2)
    pts_t = cv2.perspectiveTransform(pts, H_dict[i])
    total_corner_pts.append(pts_t)


pts = np.concatenate(total_corner_pts, axis=0)


[xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
[xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
# print(xmin, ymin, xmax, ymax)

Ht = np.reshape(np.array([1, 0, -xmin, 0, 1, -ymin, 0, 0, 1]), (3, 3))

dsize = (xmax - xmin, ymax - ymin)

result = np.zeros((ymax - ymin, xmax - xmin, 3), dtype=np.int)
print("Panorama Dimension: {}, {}".format(dsize[0], dsize[1]))
print("Stitching All images into panorama and blending")
count = 0
for i in order_list:
    count += 1
    print("Stiching and blending {} out of {} images".format(count, len(order_list)))
    img3 = pics[i]
    resultn = cv2.warpPerspective(img3, Ht.dot(H_dict[i]), dsize)

    if result.sum() == 0:
        result = resultn
    else:
        result = blend_images(resultn, result)
    


print("Stitching Done")
# result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

cv2.imwrite(panoname + ".jpg", result)

# print(-ymin, -ymin + pics[i].shape[0])

# result = result[-ymin:-ymin + pics[order_list[middle]].shape[0], :, :]

# cv2.imwrite(panoname + "_cut.jpg", result)

print("Image mosaic stored as " + panoname + ".jpg")

