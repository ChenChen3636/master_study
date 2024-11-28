import cv2
import numpy as np
import os
import json

# 圖像文件的資料夾路徑
data_folder = 'data\\stitched_evaluate_data'

# 獲取資料夾內所有的文件
files = os.listdir(data_folder)

# 過濾出所有的L開頭和S開頭的圖片文件
left_images = sorted([f for f in files if f.startswith('L') and f.endswith('.png')])
right_images = sorted([f for f in files if f.startswith('S') and f.endswith('.png')])

# 確保左右圖像數量相等且編號匹配
assert len(left_images) == len(right_images), "左右圖像數量不匹配或文件名編號不對應"

# 遍歷所有圖像對
for left_image, right_image in zip(left_images, right_images):
    image_path_left = os.path.join(data_folder, left_image)
    image_path_right = os.path.join(data_folder, right_image)

    # 讀取兩張超音波圖像
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
        # 如果標籤不存在，返回 None
        return None

    coord_left = get_coord_from_json(json_path_left, 'L5')
    coord_right = get_coord_from_json(json_path_right, 'L5')

    if coord_left is None or coord_right is None:
        raise ValueError(f"未能在JSON文件中找到指定的標籤 'L5'，圖像對：{left_image}, {right_image}")

    print(f'Processing {left_image} and {right_image}')
    print(f'coord_left: {coord_left}')
    print(f'coord_right: {coord_right}')

    # 計算偏移量
    x_offset = coord_left[0] - coord_right[0] 
    y_offset = coord_left[1] - coord_right[1] 

    print(f'x_offset: {x_offset}')
    print(f'y_offset: {y_offset}')

    # 創建拼接圖像的大小
    height = max(img_left.shape[0], img_right.shape[0] + abs(y_offset))
    width = max(img_left.shape[1], img_right.shape[1] + abs(x_offset))

    print(f'image_height: {img_right.shape[0]}')
    print(f'image_width: {img_right.shape[1]}')


    print(f'new_height: {height}')
    print(f'new_width: {width}')

    stitched_image = np.zeros((height, width), dtype=np.uint8)

    # 將左圖放入新的圖像中
    stitched_image[:img_left.shape[0], :img_left.shape[1]] = img_left

    # 建立兩個權重矩陣，用於雙向加權融合
    weight_matrix_left = np.zeros((height, width), dtype=np.float32)
    weight_matrix_right = np.zeros((height, width), dtype=np.float32)

    # 計算重疊區域
    overlap_start_x = max(x_offset, 0)
    overlap_end_x = min(img_left.shape[1], img_right.shape[1] + x_offset)

    # 填充重疊區域內的權重
    for x in range(width):
        if x < overlap_start_x:  # 左邊界
            ω1 = 1.0
            ω2 = 0.0
        elif overlap_start_x <= x < overlap_end_x:  # 重疊區域內
            ω1 = (x - x_offset) / (overlap_end_x - overlap_start_x)
            ω2 = 1 - ω1
        else:  # 右邊界
            ω1 = 0.0
            ω2 = 1.0

        weight_matrix_left[:, x] = ω1
        weight_matrix_right[:, x] = ω2

    # 根據計算的平移量將右圖放入新的圖像中，並進行加權融合
    for y in range(max(0, y_offset), min(height, img_right.shape[0] + y_offset)):
        for x in range(width):
            # 確保索引在合法範圍內
            if 0 <= y < img_left.shape[0] and 0 <= x < img_left.shape[1]:
                left_value = img_left[y, x]
            else:
                left_value = 0

            if 0 <= (y - y_offset) < img_right.shape[0] and 0 <= (x - x_offset) < img_right.shape[1]:
                right_value = img_right[y - y_offset, x - x_offset]
            else:
                right_value = 0

            stitched_image[y, x] = (weight_matrix_left[y, x] * left_value + 
                                     weight_matrix_right[y, x] * right_value).astype(np.uint8)
            
    
    # 保存結果
    file_number = ''.join(filter(str.isdigit, left_image))
    saved_image_name = f'blending{file_number}.png'
    saved_file_path = f'./output/stitched_images_evaluate/{saved_image_name}'

    # 確保保存目錄存在
    os.makedirs(os.path.dirname(saved_file_path), exist_ok=True)

    cv2.imwrite(saved_file_path, stitched_image)

    print(f'Saved stitched image: {saved_file_path}')
