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
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    if type(lines) != type(None):
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imshow('Medial Axis of tool',frame)
    #cv2.waitKey(0)
    return 0

if __name__ == "__main__":
    cap = cv2.VideoCapture('Vids\\1.mp4')
    fgbg = cv2.createBackgroundSubtractorMOG2()
    ret = 1
    num_frame = 0
    while(ret):
        num_frame += 1
        ret, frame = cap.read()
        #cv2.imshow("test",frame);
        #cv2.waitKey(0);
        fgmask = fgbg.apply(frame)
        cv2.imshow('Background Subtracted',fgmask)
        #cv2.waitKey(0)

        #Noise Removal
        kernel = np.ones((5, 5), np.uint8)
        fgmask_dilate = cv2.dilate(fgmask, kernel, iterations=1)

        #EdgeDetector
        EdgeDetector(fgmask)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
