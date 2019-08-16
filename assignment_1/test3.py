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

outfilename = ".".join(filename.split(".")[:-1]) + "_3_out.avi"
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
out = cv2.VideoWriter(outfilename,fourcc, 30, (1920, 1080))

past_lines = []
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

#         fgmask_1 = cv2.erode(fgmask, kernel, iterations = 2)
#         fgmask_2 = cv2.dilate(fgmask_1, kernel, iterations = 4)
#         fgmask_3 = cv2.erode(fgmask_2, kernel, iterations = 4)

        final = fgmask

        edges = sobel(final)

        lines = cv2.HoughLinesP(edges,rho = 1,theta = 1*np.pi/180,threshold = 150,minLineLength = 30,maxLineGap = 50)

        slopes = []
        final_lines = []
        if type(lines) != type(None):
            for line in lines:
                x1,y1,x2,y2 = line[0]
                if y1 < y2:
                    line[0] = np.array([x1, y1, x2, y2])
                else:
                    line[0] = np.array([x2, y2, x1, y1])
                slopes.append(np.arctan(float(y2 - y1 + 0.001) / (x2 - x1 + 0.001)))
            hist_slope = np.histogram(slopes, bins= 20)
            mode = np.argmax(hist_slope[0])
            max_angle = [hist_slope[1][mode], hist_slope[1][mode + 1]]
#             print(hist_slope)
#             print(max_angle)
            lengths = []
            for line in lines:
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
            # print(final_lines)
            if len(final_lines) > 1:
                final_lines = np.sum(np.multiply(final_lines, lengths.reshape(-1, 1)), axis = 0) / np.sum(lengths)
                final_lines = final_lines.astype(int)
            elif len(final_lines) == 1:
                final_lines = final_lines[0]            
#             if len(final_lines) > 1:
#                 final_lines = np.mean(final_lines, axis = 0).astype(int)
#             elif len(final_lines) == 1:
#                 final_lines = final_lines[0][0]
#                 final_lines = [[final_lines[0], final_lines[1], final_lines[2], final_lines[3]]]
#             else:
#                 print("Problem: ", final_lines)
            # print(final_lines)
            
        
        
        if len(final_lines) != 0:
            x1,y1,x2,y2 = final_lines[0], final_lines[1], final_lines[2], final_lines[3]
            cv2.line(frame,(x1,y1),(x2,y2),(255, 0, 255),5)    
                
#         if len(final_lines) != 0:
#             for line in final_lines:        
#                 for x1,y1,x2,y2 in line:
#                     cv2.line(frame,(x1,y1),(x2,y2),(255, 0, 255),5)

                    
        # if number % 10 == 0 or number < 10:
        #     plt.imshow(frame)
        #     plt.show()
            
#             plt.hist(slopes)
#             plt.show()
        out.write(frame)
x.release()
out.release