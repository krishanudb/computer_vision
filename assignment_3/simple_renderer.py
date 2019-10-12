import sys 
import os 
import cv2 
from cv2 import aruco 
import numpy as np 
import matplotlib.pyplot as plt


class OBJ:
    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        material = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = map(float, values[1:4])
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = map(float, values[1:4])
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(map(float, values[1:3]))
            #elif values[0] in ('usemtl', 'usemat'):
                #material = values[1]
            #elif values[0] == 'mtllib':
                #self.mtl = MTL(values[1])
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                #self.faces.append((face, norms, texcoords, material))
                self.faces.append((face, norms, texcoords))

#Define the camera calibration matrix    
camera_matrix = np.array(
                        [[1.40285620e+03, 0.00000000e+00, 5.62382483e+02],
                         [0.00000000e+00, 1.35573035e+03, 3.02431146e+02],
                         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],dtype = "double") 

 
cap = cv2.VideoCapture('camdata/vid.mp4')

# define aruco dictionary (in our case, its 6x6x250) 
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100 ) 
markerLength = 0.25   # Here, our measurement unit is centimetre. 
arucoParams = cv2.aruco.DetectorParameters_create()  
dist_coeffs = None

obj = OBJ('Car.obj', swapyz=True)  


outfilename = "test3.avi"
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
out = cv2.VideoWriter(outfilename,fourcc, 30, (1280, 720))


ret =1
count = 0
while (ret):
    print(count)
    count += 1
    ret, frame = cap.read()
    if type(frame) == type(None):
        break
    size = frame.shape
    imgRemapped_gray = frame    
    avg1 = np.float32(imgRemapped_gray) 
    avg2 = np.float32(imgRemapped_gray) 
  
    # Detect aruco 
    aruco_corners = cv2.aruco.detectMarkers(imgRemapped_gray, aruco_dict, parameters = arucoParams)  
    imgWithAruco = imgRemapped_gray  # assigning imRemapped_color to imgWithAruco 
    if len(aruco_corners[0]) > 0: 


        x1 = (aruco_corners[0][0][0][0][0], aruco_corners[0][0][0][0][1]) 
        x2 = (aruco_corners[0][0][0][1][0], aruco_corners[0][0][0][1][1]) 
        x3 = (aruco_corners[0][0][0][2][0], aruco_corners[0][0][0][2][1]) 
        x4 = (aruco_corners[0][0][0][3][0], aruco_corners[0][0][0][3][1])


        meanx = np.mean([x1, x2, x3, x4], axis = 0)

        x1 = np.array([aruco_corners[0][0][0][0][0], aruco_corners[0][0][0][0][1]]).astype(np.float32) 
        x2 = np.array([aruco_corners[0][0][0][1][0], aruco_corners[0][0][0][1][1]]).astype(np.float32) 
        x3 = np.array([aruco_corners[0][0][0][2][0], aruco_corners[0][0][0][2][1]]).astype(np.float32) 
        x4 = np.array([aruco_corners[0][0][0][3][0], aruco_corners[0][0][0][3][1]]).astype(np.float32)
        corners = [np.array([x1, x2, x3, x4]).astype(np.float32)]
        # print(corners)
        meanx[0] = 1280/2 - meanx[0]
        meanx[1] = 720/2 - meanx[1]

        if aruco_corners[1] != None:
             
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, markerLength, camera_matrix, dist_coeffs)

            dst, jacobian = cv2.Rodrigues(rvec) 
    
            vertices = obj.vertices
        
            scale_matrix = np.eye(3) * 0.1
        #     plt.imshow(img)
        #     plt.show();
        
            for face in obj.faces:
                face_vertices = face[0]
                points = np.array([vertices[vertex - 1] for vertex in face_vertices])
                points = np.dot(points, scale_matrix)

                points = np.array([[p[0], p[1], p[2]] for p in points])
                
                imgpts, jac = cv2.projectPoints(points, rvec, tvec, camera_matrix, dist_coeffs)
                

                imgpts = np.int32(imgpts)

                cv2.fillConvexPoly(frame, imgpts, (137, 27, 211))
            
            out.write(frame)
        else:
            out.write(frame)

    else:
        out.write(frame)
        
print('All frames done!! End!!')
cap.realease()
out.realease()
