good_matches_dict = {}
MIN_MATCH_COUNT = 40
for i in kpics.keys():
    good_matches_dict[i] = {} 
    for j in kpics.keys():
        if i != j:
            good_matches_dict[i][j] = []
            for match in matches_dict[i][j]:
                if match.distance < percentile_matches:
                    good_matches_dict[i][j].append(match)
             
            good_matches = good_matches_dict[i][j]

            print("Between {} and {}, number of good matches: {}".format(i, j, len(good_matches)))
            if len(good_matches)>MIN_MATCH_COUNT:
                img1, img2, kp1, kp2 = kpics[i], kpics[j], kp[i], kp[j]
                cimg1, cimg2 = pics[i], pics[j]
                img1_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
                img2_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)                
                
                H1, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC,2)
                H2, mask = cv2.findHomography(img2_pts, img1_pts, cv2.RANSAC,2)
                
                H2_inverse = np.linalg.inv(H2)                    
                H = (H1 + H2_inverse) / 2.
#                 print('Homo', H)
                h1,w1 = img1.shape[:2]
                h2,w2 = img2.shape[:2]
                pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
                pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
#                 print('pts1',pts1)
#                 print('pts2',pts2)

                pts2_ = cv2.perspectiveTransform(pts2, H)
                pts = np.concatenate((pts1, pts2_), axis=0)
#                 print('pts2 after transform',pts2_)

                [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
                [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)

                t = [-xmin,-ymin]
#                 print('tmin',t)

                Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

                result = cv2.warpPerspective(cimg1, Ht.dot(H), (xmax-xmin, ymax-ymin))
                ret,thresh1 = cv2.threshold(result,0,255,cv2.THRESH_BINARY)
#                 plt.imshow(thresh1,'gray'),plt.show()
                result2 = cv2.warpPerspective(cimg2, Ht.dot(np.eye(3)), (xmax-xmin, ymax-ymin))
                ret,thresh2 = cv2.threshold(result2,0,255,cv2.THRESH_BINARY)
#                 plt.imshow(thresh2,'gray'),plt.show()  
                overlap = cv2.bitwise_and(thresh1,thresh2)
#                 plt.imshow(overlap,'gray'),plt.show()
                overlap = np.array(overlap.sum(axis=-1)/3. != 0, dtype=np.int8)
#                 print('final homo',Ht.dot(H))
                xmin = np.nonzero(overlap)[1].min()
                xmax = np.nonzero(overlap)[1].max()
                x_overlap = np.zeros(overlap.shape)
                for i in range(0, x_overlap.shape[1]):
                    x_overlap[:, i] = i
                no = overlap * (x_overlap - xmin) / (xmax - xmin)
                on = (1 - no) * overlap
                mask1 = no + np.array(no == 0, dtype=np.int8)
#                 plt.imshow(mask1,'gray'),plt.show()
#                 plt.imshow(result,'gray'),plt.show()  
                new_img1 = result * mask1.reshape(mask1.shape[0], mask1.shape[1], 1)
                new_img1 = new_img1.astype(np.uint8)
#                 plt.imshow(new_img1,'gray'),plt.show()
                mask2 = on + np.array(on == 0, dtype=np.int8)
#                 plt.imshow(result2,'gray'),plt.show()  
#                 plt.imshow(mask2,'gray'),plt.show()
                new_img2 = result2 * mask2.reshape(mask2.shape[0], mask2.shape[1], 1)
                new_img2 = new_img2.astype(np.uint8)
#                 plt.imshow(new_img2,'gray'),plt.show()

                print('Before alpha Blending')
                present_old = (result[t[1]:h1+t[1],t[0]:w1+t[0]] != 0).astype(np.int8)
                result[t[1]:h1+t[1],t[0]:w1+t[0]] = (cimg2 + present_old * result[t[1]:h1+t[1],t[0]:w1+t[0]])/(1 + present_old)
                result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(20, 10)),plt.imshow(result),plt.show()
                print('After alpha Blending')
                result_new = new_img1 + new_img2
                result_new = cv2.cvtColor(result_new, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(20, 10)),plt.imshow(result_new),plt.show()

                
                result = cv2.resize(result,None,fx=0.5,fy=0.5)
#                     plt.imsave(folder + "/" +  "temp1_" + i + "_" + j + ".png", result)
                print("Feature matching for images pairs {} and {} done".format(i, j))
#               Image blending
            else:
                print ('Not enough matches are found between {} and {}'.format(i, j))
    print("Matching features for image {} with all image pairs Done".format(i))
print("Feature matching done for all images")
print("End !!")
