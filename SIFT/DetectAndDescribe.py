import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

def image_read():
    # 讀取影像 1a:前半段脊椎 1b:後半段脊椎(靠近尾椎)
    
    img1 = cv2.imread('data\\LS_data\\L2.png')
    img2 = cv2.imread('data\\LS_data\\S2.png')

    return img1,img2

def cropped_img(img):

    x,y = 167,59 #剪裁座標
    width = 690 
    height = 551

    cropped_img = img[y:y+height, x:x+width]

    return cropped_img

def create_mask(img,mask_type,mask_w,distance_from_edge):
    
    #照片初始大小 768*1024 
    original_height, original_width = img.shape[:2]

    # 遮罩大小
    mask_width = mask_w #690
    mask_height = original_height
    
    # 根據遮罩類型計算水平起始位置
    if mask_type == 'right':
        horizontal_start = distance_from_edge
    elif mask_type == 'left':
        horizontal_start = original_width - mask_width - distance_from_edge
    else:
        raise ValueError("Invalid mask type. Use 'left' or 'right'.")

    # 創建與原始圖片相同大小的全黑遮罩
    mask = np.zeros((original_height, original_width), dtype=np.uint8)

    # 在遮罩上定義一個白色的矩形區域
    mask[:mask_height, horizontal_start:horizontal_start + mask_width] = 255
    
    return mask

def detect_feature(img,mask=None):
    
    # 創建SIFT對象
    sift = cv2.SIFT_create() # type: ignore

    # 檢測SIFT特徵點
    keypoints, descriptors = sift.detectAndCompute(img,mask)

    # # 在影像上畫出特徵點
    # keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
    # # 顯示圖片
    # cv2.imshow('Combined SIFT Keypoints', keypoints)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return keypoints,descriptors

def match_feature(img1, img2, keypoints1, keypoints2, descriptors1, descriptors2, threashold):
    
    # 使用BFMatcher進行匹配
    bf = cv2.BFMatcher(cv2.NORM_L2)

    # 使用knnMatch執行最佳匹配，取得每個點的前兩個最佳匹配
    matches = bf.knnMatch(descriptors1, descriptors2,k=2)
    
    #Ratio Test 提升匹配品質的技術，如果m.distance比n.distance距離還好(1-threashold)*100%，才算是有效匹配
    #例如threashold = 0.8，表示 m.distance 要比 n.distance 好 20% 
    good = []
    for m, n in matches:
        if m.distance < threashold* n.distance:
            good.append(m)
    
    #取得匹配座標點        
    matches = []
    for match in good:
        pt1 = keypoints1[match.queryIdx].pt
        pt2 = keypoints2[match.trainIdx].pt
        matches.append(pt1 + pt2)
        
    
    # 根據距離排序，距離越小表示匹配越好
    # matches = [x1,y1,x2,y2]
    matches = np.array(matches)
    return matches
    
def plot_matches(matches, total_img):
    
    #繪製匹配圖
    match_img = total_img.copy()
    offset = total_img.shape[1]/2
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(match_img).astype('uint8')) #　RGB is integer type
    
    ax.plot(matches[:, 0], matches[:, 1], 'xr')
    ax.plot(matches[:, 2] + offset, matches[:, 3], 'xr')
     
    ax.plot([matches[:, 0], matches[:, 2] + offset], [matches[:, 1], matches[:, 3]],
            'r', linewidth=0.5)

    # plt.show()
    
def homography(pairs):
    #計算Homography matrix單應性矩陣，pairs是每一組對應點(x1,y1,x2,y2)
    
    rows = [] #儲存轉換為線性方程組的對應點
    for i in range(pairs.shape[0]):
        p1 = np.append(pairs[i][0:2], 1) #從pairs中提取 img1 的點(x1, y1)，並將其擴展為(x1, y1, 1)。這是將二維坐標轉換為齊次坐標的過程。
        p2 = np.append(pairs[i][2:4], 1)
        row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*p1[2]] #將每對點的坐標轉換為單應性矩陣的線性方程形式。每對點會產生兩行方程，這些方程用於後續的矩陣運算以求解單應性矩陣。
        row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2]]
        rows.append(row1)
        rows.append(row2)
    rows = np.array(rows)
    U, s, V = np.linalg.svd(rows)
    H = V[-1].reshape(3, 3)
    H = H/H[2, 2] # standardize to let w*H[2,2] = 1
    
    return H

def random_point(matches, k=4):
    idx = random.sample(range(len(matches)), k)
    point = [matches[i] for i in idx ]
    
    return np.array(point) #回傳包含隨機選擇的匹配點數組

def get_error(points, H):
    #計算ˋ給定單應性矩陣'H'的預測誤差，points是對應點對的集合
    
    num_points = len(points)
    all_p1 = np.concatenate((points[:, 0:2], np.ones((num_points, 1))), axis=1)
    all_p2 = points[:, 2:4]
    estimate_p2 = np.zeros((num_points, 2))
    for i in range(num_points):
        temp = np.dot(H, all_p1[i])
        estimate_p2[i] = (temp/temp[2])[0:2] # set index 2 to 1 and slice the index 0, 1
    # Compute error
    errors = np.linalg.norm(all_p2 - estimate_p2 , axis=1) ** 2

    return errors #回傳每對點的誤差平方數組

def ransac(matches, threshold, iters): #matches（匹配點對的集合）、threshold（用於確定inliers的誤差閾值）和 iters（RANSAC算法的迭代次數)
    num_best_inliers = 0
    
    for i in range(iters):
        points = random_point(matches)
        H = homography(points)
        
        #  avoid dividing by zero 
        if np.linalg.matrix_rank(H) < 3:
            continue
            
        errors = get_error(matches, H)
        idx = np.where(errors < threshold)[0]
        inliers = matches[idx]

        num_inliers = len(inliers)
        if num_inliers > num_best_inliers:
            best_inliers = inliers.copy()
            num_best_inliers = num_inliers
            best_H = H.copy()
            
    print("inliers/matches: {}/{}".format(num_best_inliers, len(matches)))
    return best_inliers, best_H

def stitch_img(img1, img2, H):
    # Convert to double and normalize. Avoid noise.
    left = cv2.normalize(img1.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)   
    # Convert to double and normalize.
    right = cv2.normalize(img2.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)   
    
    # left image
    height_l, width_l, channel_l = left.shape
    corners = [[0, 0, 1], [width_l, 0, 1], [width_l, height_l, 1], [0, height_l, 1]]
    corners_new = [np.dot(H, corner) for corner in corners]
    corners_new = np.array(corners_new).T 
    x_news = corners_new[0] / corners_new[2]
    y_news = corners_new[1] / corners_new[2]
    y_min = min(y_news)
    x_min = min(x_news)

    translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    H = np.dot(translation_mat, H)
    
    # Get height, width
    height_new = int(round(abs(y_min) + height_l))
    width_new = int(round(abs(x_min) + width_l))
    size = (width_new, height_new)

    # right image
    warped_l = cv2.warpPerspective(src=left, M=H, dsize=size)

    height_r, width_r, channel_r = right.shape
    
    height_new = int(round(abs(y_min) + height_r))
    width_new = int(round(abs(x_min) + width_r))
    size = (width_new, height_new)
    

    warped_r = cv2.warpPerspective(src=right, M=translation_mat, dsize=size)
     
    black = np.zeros(3)  # Black pixel.
    
    # Stitching procedure, store results in warped_l.
    for i in range((warped_r.shape[0])):
        for j in range(warped_r.shape[1]):
            pixel_l = warped_l[i, j, :]
            pixel_r = warped_r[i, j, :]
            
            if not np.array_equal(pixel_l, black) and np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_l
            elif np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_r
            elif not np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = (pixel_l + pixel_r) / 2
            else:
                pass
                  
    stitch_image = warped_l[:warped_r.shape[0], :warped_r.shape[1], :]
    return stitch_image

def main():
    img1, img2 = image_read() #讀取原始圖片
    img1 = cropped_img(img1) #剪裁左圖
    img2 = cropped_img(img2) #剪裁右圖
    
            
            
    #creat_mask(image,'left' or 'right', mask_width, mask from edge，左圖用距離左邊界計算，右圖用距離右邊界計算)
    # mask1 = create_mask(img1,'left',300,0) # 創建左圖遮罩 
    # mask2 = create_mask(img2,'right',260,0) #創建右圖遮罩
    # keypoints1, descriptors1 = detect_feature(img1,mask=mask1) #提取左圖特徵點
    # keypoints2, descriptors2 = detect_feature(img2,mask=mask2) #提取右圖特徵點
    # matches = match_feature(img1, img2, keypoints1, keypoints2, descriptors1, descriptors2, 0.9) #特徵點匹配
    # print(format(matches.shape[0]))
    
    # total_img = np.concatenate((img1,img2), axis=1) #將兩張圖片img1,img2沿著水平軸拼接在一起
    # plot_matches(matches, total_img) #繪製匹配結果圖
    # inliers, H = ransac(matches, 0.8, 2000)
    # plot_matches(inliers,total_img)
    # stitch_image = stitch_img(img1, img2, H)
    # plt.imshow(stitch_img(img1, img2, H))
    # plt.show()
    # Save the stitched image
    # cv2.imwrite('stitched_image.jpg', stitch_image * 255) #將圖片儲存到當前目錄下，並將像素值轉換為 0-255 的範圍
    
if __name__ == '__main__':
    main()


