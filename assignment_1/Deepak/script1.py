import numpy as np
import cv2

def EdgeDetector(path):
    window_name = ('Sobel - Simple edge detector')

    src = cv2.GaussianBlur(path, (3,3), 0)
    sobel_x = cv2.Sobel(src,cv2.CV_64F, 1, 0, ksize=5, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    sobel_y = cv2.Sobel(src,cv2.CV_64F, 0, 1, ksize=5, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)

    edges= cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

    #cv2.imshow(window_name,edges)
    #cv2.waitKey(0)

    ##-----Hough Transform-----##
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=50)

    if type(lines) != type(None):
        for x1, y1, x2, y2 in lines[0]:
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 5)
            cv2.imshow('Medial Axis of tool',frame)

    #cv2.waitKey(0)
    out.write(frame)
    return 0

if __name__ == "__main__":
    cap = cv2.VideoCapture('Vids\\1.mp4')
    fgbg = cv2.createBackgroundSubtractorMOG2()

    out = cv2.VideoWriter('1_detect.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (1920, 1080))

    ret = 1
    num_frame = 0
    while(ret):
        num_frame += 1
        ret, frame = cap.read()
        #cv2.imshow("test",frame);
        #cv2.waitKey(0);
        fgmask = fgbg.apply(frame)
        #cv2.imshow('Background Subtracted',fgmask)
        #cv2.waitKey(0)

        #Noise Removal
        if type(fgmask) != type(None):
            kernel = np.ones((5, 5), np.uint8)
            fgmask_dilate = cv2.dilate(fgmask, kernel, iterations=1)

            #EdgeDetector
            EdgeDetector(fgmask)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

