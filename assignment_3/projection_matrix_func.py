# The logic of calculating projection matrix is taken from
# https://bitesofcode.wordpress.com/2018/09/16/augmented-reality-with-python-and-opencv-part-2/
# with slight modifications
def projection_matrix(camera_params, H):
    rot_and_transl = np.dot(np.linalg.inv(camera_params), H)
    c1 = rot_and_transl[:, 0]
    c2 = rot_and_transl[:, 1]
    c3 = rot_and_transl[:, 2]
    norm = math.sqrt(np.linalg.norm(c1, 2) * np.linalg.norm(c2, 2))
    r1 = c1 / norm
    r2 = c2 / norm
    t = c3 / norm
    sum_r1_r2 = r1 + r2
    cprod_r1_r2 = np.cross(r1, r2)
    cprod = np.cross(sum_r1_r2, cprod_r1_r2)
    r1 = np.dot(sum_r1_r2 / np.linalg.norm(sum_r1_r2, 2) + cprod / np.linalg.norm(cprod, 2), 1 / math.sqrt(2))
    r2 = np.dot(sum_r1_r2 / np.linalg.norm(sum_r1_r2, 2) - cprod / np.linalg.norm(cprod, 2), 1 / math.sqrt(2))
    r3 = -np.cross(r1, r2)
    proj_matrix= np.stack((r1, r2, r3, t)).T
    return proj_matrix
