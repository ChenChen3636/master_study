import os
import csv
import cv2
import json

def process_image(input_path, output_path, writer, image_filename, json_data):
    original_image = cv2.imread(input_path)

    # 從 JSON 檔中提取 centers
    centers = [tuple(map(int, point)) for shape in json_data['shapes'] for point in shape['points']]
    
    ### 連接中心點
    for i in range(len(centers) - 1):
        cv2.line(original_image, centers[i], centers[i + 1], (0, 255, 255), 2)
    
    ### 計算兩點之間的斜率
    slopes = []
    for i in range(3):
        x1, y1 = centers[i]
        x2, y2 = centers[i + 1]
        slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')  # 避免除以零
        slopes.append(slope)
    
    ### 找出斜率變化最大的地方
    changes = []
    max_change = 0
    index_of_max_change = 0
    previous_slope = slopes[0]

    for i in range(1, 3):
        change = abs(slopes[i] - previous_slope)
        changes.append(change)
        if change > max_change:
            max_change = change
            index_of_max_change = i
        previous_slope = slopes[i]
        writer.writerow([image_filename, i, change])

    ### 繪製 keypoint
    font = cv2.FONT_HERSHEY_SIMPLEX  # 文字字體
    font_thickness = 2
    key_point = centers[index_of_max_change]
    cv2.circle(original_image, centers[index_of_max_change], 10, (0, 0, 255), -1)
    cv2.putText(original_image, 'key_point', (key_point[0] - 20, key_point[1] + 35), font, 1, (0, 0, 255), font_thickness)
    print(f"{image_filename} : {index_of_max_change}")

    ### 儲存影像
    cv2.imwrite(output_path, original_image)

if __name__ == '__main__':
    input_folder = 'data\LS_data' # type: ignore
    output_folder = 'output/S1_detection/LS_data'
    csv_file_path = os.path.join(output_folder, "slope_change.csv")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'idx', 'slope_change'])

        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.png', '.jpg')):
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, filename)
                
                # 讀取對應的 JSON 檔
                json_path = input_path.replace('.png', '.json').replace('.jpg', '.json')
                with open(json_path, 'r') as json_file:
                    json_data = json.load(json_file)

                process_image(input_path, output_path, writer, filename, json_data)
                print(f"Processed and saved {filename}")
