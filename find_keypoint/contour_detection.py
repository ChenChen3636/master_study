import cv2
import numpy as np
import os
import csv
from sklearn.cluster  import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_angle(pt1, pt2, pt3):
    #計算由三點定義的角度
    ba = np.array(pt1) - np.array(pt2)
    bc = np.array(pt3) - np.array(pt2)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0) # numpy.np(x,x_min,x_max)，將元素限制在範圍內，超過就強制等於設定值
    return np.arccos(cosine_angle) # 反余弦值: y = cos(x), x = arrcos(y)

def calculate_angles_with_basepoint(centers):
    #計算每個中心點與最左邊中心點的水平夾角
    base_point = min(centers, key=lambda x: x[0])
    angles = []
    for center in centers:
        if center == base_point:
            continue
        dx = center[0] - base_point[0]
        dy = base_point[1] - center[1]
        angle = np.degrees(np.arctan2(dy, dx))
        if angle: 
            angles.append(angle)
        
    max_change = 0
    index_of_max_change = 0
    previous_angle = angles[0]
    for i in range(1, len(angles)):
        change = angles[i] - previous_angle
        if change > max_change:
            max_change = change
            index_of_max_change = i
        previous_angle = angles[i]
        
    return base_point, angles, index_of_max_change

def calculate_distance(point1, point2):
    #計算兩點之間的距離
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.linalg.norm(point1 - point2) #歐基里德距離

def draw_distances_on_image(image, centers, writer, image_filename):
    #繪製兩點距離在圖上，用於計算破碎輪廓，找出 threshold 值
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 0, 255)
    font_thickness = 2

    for i in range(len(centers) - 1):
        point1 = centers[i]
        point2 = centers[i + 1]
        
        # 計算距離並格式化到小數點第一位
        distance = calculate_distance(point1, point2)
        formatted_distance = f"{distance:.1f}"

        # 放置文本位置
        text_position = ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)

        # 繪製中心點連線和距離
        cv2.line(image, point1, point2, (0, 255, 0), 1)
        cv2.putText(image, formatted_distance, text_position, font, font_scale, font_color, font_thickness)
        
        # 寫入csv
        writer.writerow([image_filename, i, formatted_distance])

    return image

def merge_centers(centers, threshold):
    #合併破碎的輪廓
    clusters = []
    mappings = []  # To store mapping from original to merged centers

    for point in centers:
        found = False
        for cluster in clusters:
            if calculate_distance(point, cluster['centroid']) < threshold:
                cluster['points'].append(point)
                cluster['centroid'] = tuple(np.mean(cluster['points'], axis=0).astype(int))
                found = True
                break
        if not found:
            clusters.append({'points': [point], 'centroid': point})

    # Creating mappings from original centers to their respective new centers
    for cluster in clusters:
        for point in cluster['points']:
            mappings.append((point, cluster['centroid']))

    return [cluster['centroid'] for cluster in clusters], mappings

def draw_item_on_image(image, centers, writer, image_filename):
    #在影像上繪製物件大禮包
    base_point, angles, index_of_max_change = calculate_angles_with_basepoint(centers)
    font = cv2.FONT_HERSHEY_SIMPLEX #文字字體
    font_thickness = 1
    font_color = (255, 255, 255) #文字白色
    font_scale = 0.3
    
    #-----------------------------------------------------------------------------------
    # 繪製中心點                                                                        
    #-----------------------------------------------------------------------------------
    for center in centers:
        cv2.circle(image, center, 8, (0, 255, 0), -1)  # Green color for new centers
    
    #------------------------------------------------------------------------------------    
    # 繪製XY軸 
    #------------------------------------------------------------------------------------   
    scale = 20
    num_ticks = 10 
    tick_length = 5
           
    # X軸
    cv2.line(image, (0, base_point[1]), (image.shape[1], base_point[1]), (255, 0, 0), 2)
    # Y軸
    cv2.line(image, (base_point[0], 0), (base_point[0], image.shape[0]), (0, 255, 0), 2)

    # Draw X-axis ticks and labels
    x_step = image.shape[1] // num_ticks
    for i in range(num_ticks + 1):
        x = i * x_step
        cv2.line(image, (x, base_point[1] - tick_length), (x, base_point[1] + tick_length), (255, 0, 0), 2)
        cv2.putText(image, str(x - base_point[0]), (x, base_point[1] + 20), font, font_scale, font_color, font_thickness)

    # Draw Y-axis ticks and labels
    y_step = image.shape[0] // num_ticks
    for j in range(num_ticks + 1):
        y = j * y_step
        cv2.line(image, (base_point[0] - tick_length, y), (base_point[0] + tick_length, y), (0, 255, 0), 2)
        cv2.putText(image, str(y - base_point[1]), (base_point[0] + 10, y), font, font_scale, font_color, font_thickness)    
        
    #------------------------------------------------------------------------------------    
    # 繪製key_point
    #------------------------------------------------------------------------------------  
    if centers[index_of_max_change] != base_point:
        key_point = centers[index_of_max_change]
        cv2.circle(image, centers[index_of_max_change], 8, (0 , 0, 255), -1)
        cv2.putText(image, 'key_point', (key_point[0] -10, key_point[1] + 20), font, 0.5, (0, 0, 255), font_thickness)  
    print(f"{image_filename} : {index_of_max_change}")
    #------------------------------------------------------------------------------------    
    # 繪製中心點與 base_point 的夾角角度 ====> 有錯，wait for debug!!!!
    #------------------------------------------------------------------------------------   
    angle_index = 0
    for center in centers:
        if center == base_point:
            continue
        angle = angles[angle_index]  # Use the current index for angles
        text_position = (center[0] + 20, center[1])
        cv2.putText(image, f"{angle:.1f} deg", text_position, font, font_scale, font_color, font_thickness)
        angle_index += 1

    return image

def process_image(image_path, output_path, writer, image_filename):
    
    image = cv2.imread(image_path)
    
    # Canny 邊緣檢測
    
    edges = cv2.Canny(image, 50 ,255) #(img, 最小閾值, 最大閾值)
    
    #找到輪廓並計算中心點
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.RETR_EXTERNAL : 輪廓檢索模式，這裡指檢索最外層的輪廓
    # cv2.CHAIN_APPROX_SIMPLE : 輪廓近似方法。壓縮水平、垂直和對角線段，只保留他們的端點
        
    centers = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centers.append((cx,cy))
            # cv2.circle(image, (cx,cy), 5, (255,0,0), -1)
            
    # cv2.imshow(image_filename, edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows

    # 按照x軸座標排序中心點
    centers = sorted(centers, key=lambda x: x[0])
    # print(centers)
    
    #繪製破碎中心點的距離
    # image_with_distances = draw_distances_on_image(image, centers, writer, image_filename)
    # cv2.imwrite(output_path, image_with_distances)
    
    # 合併中心點
    merged_centers, mappings = merge_centers(centers, threshold = 60) # <= 調整中心點合併閾值
    # cv2.imwrite(output_path, image_with_centers)
    
    # 連接合併中心點        
    for i in range(len(merged_centers)-1):
        cv2.line(image, merged_centers[i], merged_centers[i+1], (0,255,0), 2) 
    
    # for  original, merged in mappings:
    #     print(f"Original: {original} -> Merged: {merged}")
    
    
    #baseline劃一條水平線，比較與每個椎體之夾角，找出角度變化最大的椎體
    base_point, angles, max_change_index = calculate_angles_with_basepoint(merged_centers)
    
    # 在影像中標記 base_point
    if base_point != (-1, -1):  #確保 base_point 是有效的
        cv2.circle(image, base_point, 12, (0, 150, 255), -1)  # 使用橘色標註base_point
        
    cv2.imwrite(output_path, image)
    
    # for idx, angle in enumerate(angles, start=1):
        # writer.writerow([image_filename, idx, angle])
    
    # 從基點到中心點繪製連線
    for center in merged_centers:
        if center != base_point:
            cv2.line(image, base_point, center, (0, 255, 255), 1)  # Cyan lines to other centers
        
    image = draw_item_on_image(image, merged_centers, writer, image_filename)
    
    writer.writerow([image_filename, merged_centers[max_change_index]])
    
    cv2.imwrite(output_path, image)
    
    """
    k-means 
    """
    # 計算每個中心點與base_point的斜率
    # slopes = []
    
    ### solution 1 : basepoint連接center斜率分群
    # base_x, base_y = base_point
    # for x, y in merged_centers:
    #     if (x - base_x) != 0:  # 避免除以0的情況
    #         slope = (y - base_y) / (x - base_x)
    #     else:
    #         continue
    #     slopes.append(slope)
    
    ### solution 2 : 連接各兩點center斜率分群
    # for i in range(len(merged_centers) - 1):
    #     x1, y1 = merged_centers[i]
    #     x2, y2 = merged_centers[i+1]
    #     if (x2 - x1) != 0:
    #         slope = (y2 - y1) / (x2 - x1)
    #     else:
    #         continue  # 用一個大數值替代無窮大斜率
    #     slopes.append(slope)
    
    # print(slopes)

    # 應用KMeans算法
    # slopes_array = np.array(slopes).reshape(-1, 1)
    # kmeans = KMeans(n_clusters=2)
    # kmeans.fit(slopes_array)
    # labels = kmeans.labels_
    
    # # 可以用不同顏色標記不同群組的中心點
    # colors = [(43, 210, 254), (101, 36, 236)]  # 綠色和藍色分別代表兩個群組
    # for label, center in zip(labels, merged_centers):
    #     cv2.circle(image, center, 12, colors[label], -1)

    # cv2.imwrite(output_path, image)
    
    
if __name__ == '__main__':

    input_folder = 'data\\training_data\SegmentationClass'
    output_folder = '.\output\S1_detection\\revise_key_point'
    csv_file_path = os.path.join(output_folder, "key_point.csv")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'Key_point'])
        
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.png', '.jpg')):
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, filename)
                
                process_image(input_path, output_path, writer, filename)
                # print("processed and saved")

        # # # 顯示結果
        # print(f"最小轉折角度: {min_angle} 度, 在點 {centers[min_index]}")    
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows
                
        
        