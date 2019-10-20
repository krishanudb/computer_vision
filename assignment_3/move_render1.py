"""Input Format:
python renderer1.py marker_locaton/filename marker2_location/filename video_location/filename object_location/filename output_locaton/filename feature_detector (surf/sift)

Output: stored in ouotput_location/filename.

Program will take some time to run dependeing upon your choice for video length and complexity of object"""



import sys 
import os 
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import math
import pygame
from OpenGL.GL import *

marker1_location = sys.argv[1]
marker2_location = sys.argv[2]
video_location = sys.argv[3]
object_location = sys.argv[4]
output_filename = sys.argv[5] 
feature_detector = sys.argv[6]
if feature_detector == "surf" or feature_detector == "SURF":
    feature_detector = "surf"

print(marker2_location)
print(marker1_location)
print(video_location)
print(object_location)

camera_matrix = np.array([[2.84858472e+03, 0.00000000e+00, 9.11046976e+02],
       [0.00000000e+00, 2.40952100e+03, 3.59385332e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)


def compute_homography(desc_frame, kpt_frame, desc_model, kpt_model, M_old):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
            
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    
    matches = flann.knnMatch(desc_model,desc_frame,k=2)
    matchesMask = [[0,0] for _ in range(len(matches))]
    good_matches = []
    for m,n in matches:
        if m.distance < 0.55 * n.distance:
            good_matches.append(m)
    matches = good_matches        
    matches = sorted(matches, key = lambda x:x.distance)

    if len(matches) > 100:
        matches = matches[:100]
        
    if len(matches) >= 6:
        src_pts = np.float32([kpt_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpt_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0, maxIters=2000)
        if type(M) != type(None) and type(M_old) != type(None):

            M = M_old * 0.8 + M * 0.2
            M_old = M
        elif type(M) == type(None):
            M = M_old
        else:
            M_old = M
    else:
        M = M_old
    return matches, M


def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
#     print(hex_color)
#     hex_color = hex_color.lstrip('#')
#     h_len = len(hex_color)
#     return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))
    return colors[hex_color]

def projection_matrix(camera_parameters, homography):
    """
     From the camera calibration matrix and the estimated homography
     compute the 3D projection matrix
     """
    # Compute rotation along the x and y axis as well as the translation
#     homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = -np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return projection



def MTL(filename):
    contents = {}
    mtl = None
#     print(filename)
    for line in open(filename, "r"):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue
        if values[0] == 'newmtl':
            mtl = contents[values[1]] = {}
        elif mtl is None:
            raise ValueError, "mtl file doesn't start with newmtl stmt"
        elif values[0] == 'map_Kd':
            # load the texture referred to by this declaration
            mtl[values[0]] = values[1]
            surf = pygame.image.load(mtl['map_Kd'])
            image = pygame.image.tostring(surf, 'RGBA', 1)
            ix, iy = surf.get_rect().size
            texid = mtl['texture_Kd'] = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texid)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA,
                GL_UNSIGNED_BYTE, image)
        else:
            mtl[values[0]] = map(float, values[1:])
    return contents

class OBJ:
    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        foldername = "/".join(filename.split("/")[:-1]) + "/"
#         print(foldername)
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
            elif values[0] in ('usemtl', 'usemat'):
                material = values[1]
            elif values[0] == 'mtllib':
#                 print(values)
                self.mtl = MTL(foldername + values[1])
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
                self.faces.append((face, norms, texcoords, material))

        self.gl_list = glGenLists(1)
        glNewList(self.gl_list, GL_COMPILE)
        glEnable(GL_TEXTURE_2D)
        glFrontFace(GL_CCW)
        for face in self.faces:
            vertices, normals, texture_coords, material = face

            mtl = self.mtl[material]
            if 'texture_Kd' in mtl:
                # use diffuse texmap
                glBindTexture(GL_TEXTURE_2D, mtl['texture_Kd'])
            else:
                # just use diffuse colour
                glColor(*mtl['Kd'])

            glBegin(GL_POLYGON)
            for i in range(len(vertices)):
                if normals[i] > 0:
                    glNormal3fv(self.normals[normals[i] - 1])
                if texture_coords[i] > 0:
                    glTexCoord2fv(self.texcoords[texture_coords[i] - 1])
                glVertex3fv(self.vertices[vertices[i] - 1])
            glEnd()
        glDisable(GL_TEXTURE_2D)
        glEndList()


def render(img, obj, M1, M2, model1, model2, camera_matrix, t, pos_old, color=False):
# def render(img, obj, projection, model, t, color=False):
    
    projection2 = np.dot(camera_matrix, M2)
    projection1 = np.dot(camera_matrix, M1)

    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    
    h1, w1, _ = model1.shape
    h2, w2, _ = model2.shape
    
    cpts1 = np.array([[0. + w1 / 2., 0. + h1 / 2., 0.]]).reshape(-1, 1, 3)
    cpts2 = np.array([[0. + w2 / 2., 0. + h2 / 2., 0.]]).reshape(-1, 1, 3)

    co1 = cv2.perspectiveTransform(cpts1, projection1)
    co2 = cv2.perspectiveTransform(cpts2, projection2)

    cdisp = co1 - co2 - pos_old
    speed = cdisp / 20
    
    pos = speed + pos_old

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)

        points = np.array([[p[0] + w2 / 2, p[1] + h2 / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection2)
        dst += pos
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1] # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img, pos


model_image1 = cv2.imread(marker1_location)
model_image2 = cv2.imread(marker2_location)

model_image_binary1 = model_image1.copy()
model_image_binary2 = model_image2.copy()

model_image_binary1 = cv2.cvtColor(model_image1, cv2.COLOR_BGR2GRAY)
model_image_binary2 = cv2.cvtColor(model_image2, cv2.COLOR_BGR2GRAY)


# sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create(400)
kp_model1, des_model1 = surf.detectAndCompute(model_image_binary1, None)

# sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create(400)
kp_model2, des_model2 = surf.detectAndCompute(model_image_binary2, None)



cap = cv2.VideoCapture(video_location)

fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
out = cv2.VideoWriter(output_filename, fourcc, 30, (1920, 1080))

ret = True
number = 0

obj = OBJ(object_location, swapyz=True)

colors = {}
for part in obj.mtl.keys():
    colors[part] = (np.random.randint(255), np.random.randint(255), np.random.randint(255))


M1_old = None
M2_old = None
pos_old = np.array([0, 0])
while(ret):
    ret, frame = cap.read()
    if not ret:
        break

    frame_binary = frame.copy()
    
    frame_binary = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     _, frame_binary = cv2.threshold(frame_binary, 127,255,cv2.THRESH_BINARY)

    sift = cv2.xfeatures2d.SURF_create(400)
    kp_frame, des_frame = surf.detectAndCompute(frame_binary, None)
    
    matches1, M1 = compute_homography(des_frame, kp_frame, des_model1, kp_model1, M1_old)
    M1_old = M1
    matches2, M2 = compute_homography(des_frame, kp_frame, des_model2, kp_model2, M2_old)
    M2_old = M2
        

#     if number % 10 == 0:
    h, w, _ = model_image1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

    dst = cv2.perspectiveTransform(pts, M1)  

    temp2 = cv2.fillPoly(frame, [np.int32(dst)], [255, 0, 0]) 

    h, w, _ = model_image2.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

    dst = cv2.perspectiveTransform(pts, M2)
    
    temp2 = cv2.fillPoly(frame, [np.int32(dst)], [0, 255, 0]) 

    M1 = projection_matrix(camera_matrix, M1)  
    M2 = projection_matrix(camera_matrix, M2)  
    
    frame, pos_old = render(frame, obj, M1, M2, model_image2, model_image2, camera_matrix, number, pos_old, True)
    out.write(frame)

    number += 1
    
out.release()

print("Finished")