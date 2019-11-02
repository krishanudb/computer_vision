from __future__ import print_function


#SYNTAX: python makedata.py foldername background_subtractor(true/false) outfile

# A Numpy array will be saved in the outfile 

# Code for accessing the outfile numpy arrays: 
"""
CODE:
loaded = np.load(outfile)
images = loaded["image"]
annotations = loaded["ann"]

"""


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

ann_dict = {"stop": 1, "next": 2, "back": 3, "previous": 3, "none": 4}

def read_video(filename, typ, background_subtraction = False):
    if background_subtraction:
        bgex = cv2.createBackgroundSubtractorMOG2()
    x = cv2.VideoCapture(filename)
    ret = True
    num = 0
    im_list = []
    ann_list = []
    while(ret):
        ret, frame = x.read()
        if background_subtraction:
            fgmask = bgex.apply(frame)
        if not ret:
            break
        if num >= 15 and num % 5 == 0:
            frame = cv2.resize(frame, (50, 50))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if background_subtraction:
                fgmask = cv2.resize(fgmask, (50, 50)).reshape((50, 50, 1))
                frame = np.concatenate((frame, fgmask), axis = 2)
            im_list.append(frame)
            ann_list.append(ann_dict[typ])
        num += 1
    return im_list[:-5], ann_list[:-5]


def read_dataset(foldername, bgsub = False):
    image_dataset = []
    annotations_dataset = []
    for f in os.listdir(foldername):
        print("Doing {}".format(f))
        for fi in os.listdir(foldername + "/" + f):
#             print("Doing " + (fi), end="\r")
            tmp_ids, tmp_ads = read_video(foldername + "/" + f + "/" + fi, f, bgsub)
            image_dataset.extend(tmp_ids)
            annotations_dataset.extend(tmp_ads)
#         print("\n")
        print("Done {}".format(f))
    image_dataset, annotations_dataset = np.array(image_dataset), np.array(annotations_dataset)
    np.savez_compressed(outfile, image = image_dataset, ann = annotations_dataset)

foldernm = sys.argv[1]
background_subtract = sys.argv[2]
if background_subtract == "true" or background_subtract == "True":
    background_subtract = True
else:
    background_subtract = False

outfile = sys.argv[3]

read_dataset(foldernm, background_subtract)
