import cv2
import numpy as np
import os
import json

# 圖像文件的路徑
data_folder = 'data/stitched_evaluate_data'
files = os.listdir(data_folder)

left_images = sorted([f for f in files if f.startswith('L') and f.endswith('.png')])
right_images = sorted([f for f in files if f.startswith('S') and f.endswith('.png')])

assert len(left_images) == len(right_images), "左右影像編號不匹配"

# 遍歷所有影像
for left_image, right_image in zip(left_images, right_images):
    number = ''.join(filter(str.isdigit, left_image))
    image_path_left = os.path.join(data_folder, left_image)
    image_path_right = os.path.join(data_folder, right_image)
    
    # 讀取兩張超音波影像
    img_left = cv2.imread(image_path_left, cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(image_path_right, cv2.IMREAD_GRAYSCALE)
    
    json_path_left = os.path.splitext(image_path_left)[0] + '.json'
    json_path_right = os.path.splitext(image_path_right)[0] + '.json'

    def get_coord_from_json(json_path, label):
        with open(json_path, 'r') as f:
            data = json.load(f)
            for shape in data['shapes']:
                if shape['label'] == label:
                    return [int(shape['points'][0][0]), int(shape['points'][0][1])]
        return None

    coord_left = get_coord_from_json(json_path_left, 'L5')
    coord_right = get_coord_from_json(json_path_right, 'L5')

    if coord_left is None or coord_right is None:
        raise ValueError("未能在JSON文件中找到指定的標籤 'L5'")

    print(f'Processing {left_image} and {right_image}')
    print(f'coord_left: {coord_left}')
    print(f'coord_right: {coord_right}')

    # calculate offset
    x_offset = coord_left[0] - coord_right[0] # type: ignore
    y_offset = coord_left[1] - coord_right[1] # type: ignore

    # 創建拼接圖像的大小
    height = max(img_left.shape[0], img_right.shape[0] + abs(y_offset))
    width = max(img_left.shape[1], img_right.shape[1] + abs(x_offset))

    stitched_image = np.zeros((height,width), dtype = np.uint8)

    # 將左圖放入新的圖像中
    stitched_image[:img_left.shape[0], :img_left.shape[1]] = img_left

    # 根據計算的平移量將右圖放入新的圖像中
    if x_offset >= 0 and y_offset >= 0:
        stitched_image[y_offset:y_offset + img_right.shape[0], x_offset:x_offset + img_right.shape[1]] = img_right
    elif x_offset >= 0 and y_offset < 0:
        stitched_image[:img_right.shape[0] + y_offset, x_offset:x_offset + img_right.shape[1]] = img_right[-y_offset:, :]
        
    # 漸層高斯模糊參數
    num_steps = 3  # 漸層的步數
    initial_blur = 9  # 初始模糊強度

    if x_offset > 0:
        for step in range(num_steps):
            # 計算當前步驟的模糊內核大小，隨著步數減小
            blur_strength = initial_blur - step * (initial_blur // num_steps)
            if blur_strength % 2 == 0:
                blur_strength += 1  # 確保內核大小為奇數

            if step == 0:
                # 第一步：橫跨拼接線的模糊處理
                left_edge = max(0, x_offset - ((step + 1) * num_steps))
                right_edge = min(width, x_offset + ((step + 1) * num_steps))
                
                stitched_image[:, left_edge:right_edge] = cv2.GaussianBlur(
                    stitched_image[:, left_edge:right_edge], 
                    (blur_strength, blur_strength), 
                    0
                )
            else:
                # 後續步驟：分別處理左右邊界
                # 處理左邊界
                left_start = max(0, x_offset - (step + 1) * num_steps)
                left_end = x_offset - step * num_steps
                
                stitched_image[:, left_start:left_end] = cv2.GaussianBlur(
                    stitched_image[:, left_start:left_end], 
                    (blur_strength, blur_strength), 
                    0
                )

                # 處理右邊界
                right_start = x_offset + ((step) * num_steps)
                right_end = min(width, x_offset + ((step + 1) * num_steps))
                
                stitched_image[:, right_start:right_end] = cv2.GaussianBlur(
                    stitched_image[:, right_start:right_end], 
                    (blur_strength, blur_strength), 
                    0
                )
    
    # 保存結果
    saved_image_name = f'p2p+blur_{number}.png'
    saved_file_path = f'output\\stitched_images_evaluate/{saved_image_name}'
    
    # 確保保存目錄存在
    os.makedirs(os.path.dirname(saved_file_path), exist_ok=True)
    cv2.imwrite(saved_file_path, stitched_image)
    print(f"Saved stitched image: {saved_file_path}")
    
print("All images processed")