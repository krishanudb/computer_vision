import numpy as np
import cv2
import sys

filename = sys.argv[1]


def sobel(image):
    sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)
        
    edges = np.sqrt(sobelx ** 2 + sobely ** 2)
    edges = np.uint8(edges)
    return edges

x = cv2.VideoCapture(filename)
bgex = cv2.createBackgroundSubtractorMOG2()
ret = True
number = 0

outfilename = ".".join(filename.split(".")[:-1]) + "_out.avi"
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
out = cv2.VideoWriter(outfilename,fourcc, 30, (1920, 1080))

while(ret):
    number += 1
    ret, frame = x.read()
    fgmask = bgex.apply(frame)
    if type(fgmask) != type(None):
        kernel = np.ones((5,5),np.uint8)
        fgmask_dilate = cv2.dilate(fgmask, kernel, iterations = 1)
        fgmask_dilate_blur = cv2.blur(fgmask_dilate,(3,3))
        edges = sobel(fgmask_dilate_blur)
        
        lines = cv2.HoughLinesP(edges,rho = 1,theta = 1*np.pi/180,threshold = 100,minLineLength = 50,maxLineGap = 50)
        
        if type(lines) != type(None): 
            comb_lines = lines

            if number == 2:
                line = comb_lines[0]
                old_line = line
            else:
                min_dist = 100000
                for i in range(len(comb_lines)):
                    dist = np.sqrt(np.sum(np.square(line - comb_lines[i])))
                    if dist < min_dist:
                        min_dist = dist
                        line = comb_lines[i]
                old_line = line
                
        if type(lines) != type(None):
            for x1,y1,x2,y2 in line:
                cv2.line(frame,(x1,y1),(x2,y2),(255, 0, 255),5)
            old_lines = line
        else:
            if number != 1:
                for x1,y1,x2,y2 in old_line:
                    cv2.line(frame,(x1,y1),(x2,y2),(255, 0, 255),5)
        
        out.write(frame)
x.release()
out.release()