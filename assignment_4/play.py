# HOW TO USE THIS PROGRAM
# OPEN A NEW FOLDER AND PUT THIS PROGRAM IN THAT FOLDER
# RUN THIS PROGRAM BY PYTHON PLAY.PY
# ENTER THE CLASS OF DATA YOU WANT TO SHOOT. SHOOT AT DIFFERENT DISTANCES AT DIFFERENT LOCATIONS OF THE WEBCAM TO MAKE A VARIABLE DATASET
# WHEN YOU ARE DONE, PRESS ESC BUTTON TO STOP THE PROGRAM
# DATA WILL BE STORED IN THE FOLLOWING WAY. FOR EVERY TIME YOU RUN THE PROGRAM, A NEW FOLDER WILL BE CREATED IN A FOLDER CALLED RESULTS
# INSIDE THE NEW FOLDER, 2 NEW FOLDERS SUBTRACTION AND PIC WILL STORE THE BACKGROUND SUBTRACTED PICS AND THE RGB PICS RESPECTIVELY
# THE ANNOTATIONS.TXT FILE WILL CONTAIN THE ANNOTATIONS FOR ALL THE PICS IN THAT VIDEO.


"""CLASSES=== 0: NONE; 1: STOP; 2: NEXT; 3: PREVIOUS"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

import os

# folder_name = int(input("Enter Folder Number: "))

# print("Enter Class: ")
f = open("annotations.txt", "a")

if "results" not in os.listdir("."):
	os.mkdir("results")
folders = os.listdir("results")
if len(folders) > 0:
	dirs = [int(direc) for direc in folders]
	folder_name = max(dirs) + 1
else:
	folder_name = 0

os.mkdir("results/" + str(folder_name))
os.mkdir("results/" + str(folder_name) + "/pic")
os.mkdir("results/" + str(folder_name) + "/subtraction")

nn = 0

def capture():
	clss = int(input("Enter Class: "))
	f.write(str(folder_name) + "\t" + str(clss) + "\n")
	num = 0
	x = cv2.VideoCapture(0)
	ret = True
	bgex = cv2.createBackgroundSubtractorMOG2()

	while(ret):
	    num += 1
	    ret, frame = x.read()
	    frame = frame[50:450, 150:550, :]
	    image = frame.copy()
	    fgmask = bgex.apply(frame)
	    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	    frame = cv2.resize(frame, (600, 600))
	    fgmask = cv2.resize(fgmask, (400, 400))
	    cv2.imshow("image", frame)
	    k = cv2.waitKey(10)


	    if k == 27:
	    	x.release()
	    	cv2.destroyAllWindows()
	    	break
	    if num >= 60 and int(num) % 15 == 0:
	    	store(image, fgmask, num)
def store(img, bgsub, num):
	cv2.imwrite("results/" + str(folder_name) + "/subtraction/" + str(folder_name) + "_" +str(num) + ".jpg", bgsub)
	cv2.imwrite("results/" + str(folder_name) + "/pic/" + str(folder_name) + "_" +str(num) + ".jpg", img)

capture()