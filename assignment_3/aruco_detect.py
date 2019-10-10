#Aruco marker detection in video
import sys 
import os 
import cv2 
from cv2 import aruco 
import numpy as np 
import matplotlib.pyplot as plt
  
#Define the camera calibration matrix    
camera_matrix = np.array(
                        [[1.40285620e+03, 0.00000000e+00, 5.62382483e+02],
                         [0.00000000e+00, 1.35573035e+03, 3.02431146e+02],
                         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],dtype = "double") 

 
cap = cv2.VideoCapture('../input/camdata/vid.mp4')

# define aruco dictionary (in our case, its 6x6x250) 
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100 ) 
markerLength = 0.25   # Here, our measurement unit is centimetre. 
arucoParams = cv2.aruco.DetectorParameters_create()  
dist_coeffs = None

ret =1  
while (ret): 
    ret, frame = cap.read()
    size = frame.shape
    imgRemapped_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
    avg1 = np.float32(imgRemapped_gray) 
    avg2 = np.float32(imgRemapped_gray) 
  
    # Detect aruco 
    aruco_corners = cv2.aruco.detectMarkers(imgRemapped_gray, aruco_dict, parameters = arucoParams)  
    imgWithAruco = imgRemapped_gray  # assigning imRemapped_color to imgWithAruco 
    if len(aruco_corners[0]) > 0: 
        #print(res[0]) 
        # Drawing corners on frame 
        x1 = (aruco_corners[0][0][0][0][0], aruco_corners[0][0][0][0][1]) 
        x2 = (aruco_corners[0][0][0][1][0], aruco_corners[0][0][0][1][1]) 
        x3 = (aruco_corners[0][0][0][2][0], aruco_corners[0][0][0][2][1]) 
        x4 = (aruco_corners[0][0][0][3][0], aruco_corners[0][0][0][3][1]) 
  
        cv2.line(imgWithAruco, x1, x2, (255, 0, 0), 2) 
        cv2.line(imgWithAruco, x2, x3, (255, 0, 0), 2) 
        cv2.line(imgWithAruco, x3, x4, (255, 0, 0), 2) 
        cv2.line(imgWithAruco, x4, x1, (255, 0, 0), 2) 

        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(imgWithAruco, 'Corner 1', x1, font, 1, (255, 255, 255), 2, cv2.LINE_AA) 
        cv2.putText(imgWithAruco, 'Corner 2', x2, font, 1, (255, 255, 255), 2, cv2.LINE_AA) 
        cv2.putText(imgWithAruco, 'Corner 3', x3, font, 1, (255, 255, 255), 2, cv2.LINE_AA) 
        cv2.putText(imgWithAruco, 'Corner 4', x4, font, 1, (255, 255, 255), 2, cv2.LINE_AA) 

        if aruco_corners[1] != None:
            im_src = imgWithAruco 
            im_dst = imgWithAruco 
            pts_dst = np.array([[aruco_corners[0][0][0][0][0], aruco_corners[0][0][0][0][1]], [aruco_corners[0][0][0][1][0], aruco_corners[0][0][0][1][1]], [aruco_corners[0][0][0][2][0], aruco_corners[0][0][0][2][1]], [aruco_corners[0][0][0][3][0], aruco_corners[0][0][0][3][1]]]) 
            pts_src = pts_dst 
            H, status = cv2.findHomography(pts_src, pts_dst) 
            imgWithAruco = cv2.warpPerspective(im_src, H, (im_dst.shape[1], im_dst.shape[0])) 
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(aruco_corners[0], markerLength, camera_matrix, dist_coeffs) 
            imgWithAruco = cv2.aruco.drawAxis(imgWithAruco, camera_matrix, dist_coeffs, rvec, tvec, 10) 
            H1 = H[:, 0] 
            H2 = H[:, 1] 
            H3 = np.cross(H1, H2) 
            norm1 = np.linalg.norm(H1) 
            norm2 = np.linalg.norm(H2) 
            tnorm = (norm1 + norm2) / 2.0; 
            T = H[:, 2] / tnorm 
            cameraPose=np.mat([H1, H2, H3, T]) 
#             print(cameraPose)
            plt.imshow(imgWithAruco,'gray'), plt.show()
    else:
        print('Marker not detected')
        
print('All frames done!! End!!')
