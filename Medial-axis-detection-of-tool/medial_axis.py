# SYNTAX TO RUN THE PROGRAM
# Command: python medial_axis.py input_filename
# Output: result + input_filename (- extension) + ".avi"

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

outfilename = "result_" + filename.split(".")[0] + ".avi"
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
out = cv2.VideoWriter(outfilename,fourcc, 30, (1920, 1080))

prev_line = None

not_found = 0
while(ret):
    number += 1
    ret, frame = x.read()
    fgmask = bgex.apply(frame)
    if type(fgmask) != type(None):

        kernel = np.ones((3,3),np.uint8)
        
        fgmask_blur = cv2.blur(fgmask, (5, 5))
        
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

        final = fgmask

        edges = sobel(final)

        lines = cv2.HoughLinesP(edges,rho = 1,theta = 1*np.pi/180,threshold = 150,minLineLength = 30,maxLineGap = 50)

        slopes = []
        final_lines = []
        if type(lines) != type(None):

            cutoff_lines = []
            for line in lines:
                x1,y1,x2,y2 = line[0]
                if y1 < y2:
                    line[0] = np.array([x1, y1, x2, y2])
                else:
                    line[0] = np.array([x2, y2, x1, y1])
                if line[0][1] <= 200:
                    cutoff_lines.append(line)
                    slopes.append(np.arctan(float(y2 - y1 + 0.001) / (x2 - x1 + 0.001)))
            hist_slope = np.histogram(slopes, bins= 20)
            mode = np.argmax(hist_slope[0])
            max_angle = [hist_slope[1][mode], hist_slope[1][mode + 1]]

            lengths = []
            if len(cutoff_lines) > 0:
                for line in cutoff_lines:
                    x1,y1,x2,y2 = line[0]
                    if y1 < y2:
                        line[0] = np.array([x1, y1, x2, y2])
                    else:
                        line[0] = np.array([x2, y2, x1, y1])
                    slope = np.arctan(float(y2 - y1 + 0.001) / (x2 - x1 + 0.001))
                    if slope >= max_angle[0] and slope <= max_angle[1]:
                        final_lines.append(np.array([x1, y1, x2, y2]))
                        lengths.append(np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
                lengths = np.array(lengths)
                final_lines = np.array(final_lines)

            if len(final_lines) > 1:
                y1 = np.min(final_lines[:, 1])
                final_lines = np.sum(np.multiply(final_lines, lengths.reshape(-1, 1)), axis = 0) / np.sum(lengths)
                final_lines[1] = y1
                final_lines = final_lines.astype(int)
            elif len(final_lines) == 1:
                final_lines = final_lines[0]
        
        if len(final_lines) != 0:
            if type(prev_line) != type(None):
                final_lines = (final_lines + prev_line) / 2
            x1,y1,x2,y2 = final_lines[0], final_lines[1], final_lines[2], final_lines[3]
            prev_line = final_lines
            cv2.line(frame,(x1,y1),(x2,y2),(255, 0, 255),5)    
        else:
            not_found += 1
            if type(prev_line) != type(None):
                if not_found < 15:
                    x1,y1,x2,y2 = prev_line[0], prev_line[1], prev_line[2], prev_line[3]
                    cv2.line(frame,(x1,y1),(x2,y2),(255, 0, 255),5)
                else:
                    prev_line = None
        out.write(frame)
x.release()
out.release()