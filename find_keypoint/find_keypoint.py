import cv2
import numpy as np
import os
import csv
from sklearn.cluster  import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def process_image(image_path, output_path, writer, image_filename): 
    
    ### cv2讀取影像
    image = cv2.imread(image_path)
    
    ### 複製原始影像
    original_image = image.copy()
    
    ### 影像轉成灰階
    processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    ### 控制對比度    
    alpha = 10  
    processed_image = cv2.convertScaleAbs(processed_image, alpha=alpha)
    
    ### 影像模糊化，去除雜訊
    processed_image = cv2.medianBlur(processed_image, 7)
    
    ### Canny 邊緣檢測
    edges = cv2.Canny(processed_image, 50, 255) #(img, 最小閾值, 最大閾值)
   
    ### 找到輪廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ### cv2.RETR_EXTERNAL : 輪廓檢索模式，這裡指檢索最外層的輪廓
    ### cv2.CHAIN_APPROX_SIMPLE : 輪廓近似方法。壓縮水平、垂直和對角線段，只保留他們的端點
    
    ### 計算重心當中心點
    centers = []
    
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centers.append((cx,cy))
            cv2.circle(original_image, (cx,cy), 10, (0,255,0), -1)

    ### 按照x軸座標排序中心點
    centers = sorted(centers, key=lambda x: x[0])
    
    ### 連接中心點
    for i in range(len(centers)-1):
        cv2.line(original_image, centers[i], centers[i+1], (0,255,255), 2)
            
    ### 計算兩點之間斜率
    slopes = []
    
    for i in range(3):
        x1, y1 = centers[i]
        x2, y2 = centers[i+1]
        slope = (y2 - y1) / (x2 - x1)
        slopes.append(slope)
    
    ### 寫入 slopes 到 csv
    # for idx, slope in enumerate(slopes, start=1):
    #     writer.writerow([image_filename, idx, slope])   
    
    ### 找出斜率變化最大的地方 
    changes = []   
    max_change = 0
    index_of_max_change = 0
    previous_slope = slopes[0]
    
    for i in range(1,3):
        change = abs(slopes[i] - previous_slope)
        changes.append(change)
        if change > max_change:
            max_change = change
            index_of_max_change = i
        previous_slope = slopes[i]
        writer.writerow([image_filename, i, change])
        
    
    ### 繪製 keypoint
    font = cv2.FONT_HERSHEY_SIMPLEX #文字字體
    font_thickness = 2
    key_point = centers[index_of_max_change]
    cv2.circle(original_image, centers[index_of_max_change], 10, (0 , 0, 255), -1)
    cv2.putText(original_image, 'key_point', (key_point[0] -20, key_point[1] + 35), font, 1, (0, 0, 255), font_thickness) 
    print(f"{image_filename} : {index_of_max_change}")
    
    ### 寫入 keypoint 到 csv
    # writer.writerow([image_filename, index_of_max_change, key_point])
    
    cv2.imwrite(output_path, original_image)
    
if __name__ == '__main__':

    input_folder = 'data\\revise_data\\SegmentationClass'
    output_folder = 'output\\S1_detection\\center_slope'
    csv_file_path = os.path.join(output_folder, "slope_chane.csv")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'idx', 'slope_change'])
        
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.png', '.jpg')):
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, filename)
                
                process_image(input_path, output_path, writer, filename)
                print("processed and saved")

        ### 顯示結果
        # print(f"最小轉折角度: {min_angle} 度, 在點 {centers[min_index]}")    
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows
                
        
        