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

outfilename = ".".join(filename.split(".")[:-1]) + "_2_out.avi"
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
out = cv2.VideoWriter(outfilename,fourcc, 30, (1920, 1080))

past_lines = []
while(ret):
    number += 1
    ret, frame = x.read()
    fgmask = bgex.apply(frame)
    if type(fgmask) != type(None):
        kernel = np.ones((3,3),np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)



        final = fgmask

        edges = sobel(final)

        lines = cv2.HoughLinesP(edges,rho = 1,theta = 1*np.pi/180,threshold = 150,minLineLength = 30,maxLineGap = 50)

        if type(lines) != type(None): 
            comb_lines = lines

            if number == 2:
                line = comb_lines[0]
                x1,y1,x2,y2 = line[0]
                if y1 < y2:
                    line = np.array([[x1, y1, x2, y2]])
                else:
                    line = np.array([[x2, y2, x1, y1]])
                old_line = line
                
                past_lines.append(line)
                
            else:
                min_dist = 100000
                for i in range(len(comb_lines)):
                    x1,y1,x2,y2 = comb_lines[i][0]
                    if y1 < y2:
                        temp_line = np.array([[x1, y1, x2, y2]])
                    else:
                        temp_line = np.array([[x2, y2, x1, y1]])
                    
                    if temp_line[0][1] > 50:
                        temp_line = old_line
                    
                    dist = np.sqrt(np.sum(np.square(line - temp_line)))
                    if dist < min_dist:
                        min_dist = dist
                        line = temp_line

                old_line = line
                past_lines.append(line)
                
        if len(past_lines) > 6:
            line = ((5 * line + 3 * past_lines[-1] + 1 * past_lines[-2]  + .5 * past_lines[-3]  + .25 * past_lines[-4]) / (5 + 3 + 1 + 0.5 + 0.25)).astype(int)
            past_lines[-1] = line
            old_line = line
        elif len(past_lines) > 5:
            line = ((5 * line + 3 * past_lines[-1] + 1 * past_lines[-2]  + .5 * past_lines[-3]) / (5 + 3 + 1 + 0.5)).astype(int)
            past_lines[-1] = line
            old_line = line
        elif len(past_lines) > 4:
            line = ((5 * line + 3 * past_lines[-1] + 1 * past_lines[-2]) / (5 + 3 + 1)).astype(int)
            past_lines[-1] = line
            old_line = line
                
        if type(lines) != type(None):
            for x1,y1,x2,y2 in line:
                cv2.line(frame,(x1,y1),(x2,y2),(255, 0, 255),5)
            old_line = line
        else:
            if number != 1:
                for x1,y1,x2,y2 in old_line:
                    if np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) > 100:
                        cv2.line(frame,(x1,y1),(x2,y2),(255, 0, 255),5)
        out.write(frame)
x.release()
out.release