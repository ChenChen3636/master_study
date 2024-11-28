import cv2
import numpy as np
import os
import json

# 圖像文件的資料夾路徑
data_folder = 'data/stitched_evaluate_data/transformed_images_low'
output_folder = 'output/stitched_images_evaluate/noise_low'

# 獲取資料夾內所有的文件
files = os.listdir(data_folder)
left_images = sorted([f for f in files if f.startswith('L') and f.endswith('.png')])
right_images = sorted([f for f in files if f.startswith('noise') and f.endswith('.png')])

# 確保左右圖像數量相等且編號匹配
assert len(left_images) == len(right_images), "左右圖像數量不匹配或文件名編號不對應"

# 辨識拼接方式
def get_coord_from_json(json_path, label):
    with open(json_path, 'r') as f:
        data = json.load(f)
        for shape in data['shapes']:
            if shape['label'] == label:
                return [int(shape['points'][0][0]), int(shape['points'][0][1])]
    return None

# 拼接方式1：普通拼接
def stitch_p2p(img_left, img_right, x_offset, y_offset):
    height = max(img_left.shape[0], img_right.shape[0] + abs(y_offset))
    width = max(img_left.shape[1], img_right.shape[1] + abs(x_offset))
    stitched_image = np.zeros((height, width), dtype=np.uint8)
    stitched_image[:img_left.shape[0], :img_left.shape[1]] = img_left

    if x_offset >= 0 and y_offset >= 0:
        stitched_image[y_offset:y_offset + img_right.shape[0], x_offset:x_offset + img_right.shape[1]] = img_right
    elif x_offset >= 0 and y_offset < 0:
        stitched_image[:img_right.shape[0] + y_offset, x_offset:x_offset + img_right.shape[1]] = img_right[-y_offset:, :]

    return stitched_image

# 拼接方式2：普通拼接+模糊
def stitch_p2p_blur(img_left, img_right, x_offset, y_offset):
    stitched_image = stitch_p2p(img_left, img_right, x_offset, y_offset)

    num_steps = 3
    initial_blur = 9

    if x_offset > 0:
        for step in range(num_steps):
            blur_strength = initial_blur - step * (initial_blur // num_steps)
            if blur_strength % 2 == 0:
                blur_strength += 1

            if step == 0:
                left_edge = max(0, x_offset - ((step + 1) * num_steps))
                right_edge = min(stitched_image.shape[1], x_offset + ((step + 1) * num_steps))
                stitched_image[:, left_edge:right_edge] = cv2.GaussianBlur(
                    stitched_image[:, left_edge:right_edge], 
                    (blur_strength, blur_strength), 
                    0
                )
            else:
                left_start = max(0, x_offset - (step + 1) * num_steps)
                left_end = x_offset - step * num_steps
                stitched_image[:, left_start:left_end] = cv2.GaussianBlur(
                    stitched_image[:, left_start:left_end], 
                    (blur_strength, blur_strength), 
                    0
                )

                right_start = x_offset + ((step) * num_steps)
                right_end = min(stitched_image.shape[1], x_offset + ((step + 1) * num_steps))
                stitched_image[:, right_start:right_end] = cv2.GaussianBlur(
                    stitched_image[:, right_start:right_end], 
                    (blur_strength, blur_strength), 
                    0
                )

    return stitched_image

# 拼接方式3：加權融合
def stitch_p2p_blending(img_left, img_right, x_offset, y_offset):
    height = max(img_left.shape[0], img_right.shape[0] + abs(y_offset))
    width = max(img_left.shape[1], img_right.shape[1] + abs(x_offset))
    stitched_image = np.zeros((height, width), dtype=np.uint8)
    stitched_image[:img_left.shape[0], :img_left.shape[1]] = img_left

    weight_matrix_left = np.zeros((height, width), dtype=np.float32)
    weight_matrix_right = np.zeros((height, width), dtype=np.float32)

    overlap_start_x = max(x_offset, 0)
    overlap_end_x = min(img_left.shape[1], img_right.shape[1] + x_offset)

    for x in range(width):
        if x < overlap_start_x:
            ω1 = 1.0
            ω2 = 0.0
        elif overlap_start_x <= x < overlap_end_x:
            ω1 = (x - x_offset) / (overlap_end_x - overlap_start_x)
            ω2 = 1 - ω1
        else:
            ω1 = 0.0
            ω2 = 1.0

        weight_matrix_left[:, x] = ω1
        weight_matrix_right[:, x] = ω2

    for y in range(max(0, y_offset), min(height, img_right.shape[0] + y_offset)):
        for x in range(width):
            left_value = img_left[y, x] if 0 <= y < img_left.shape[0] and 0 <= x < img_left.shape[1] else 0
            right_value = img_right[y - y_offset, x - x_offset] if 0 <= (y - y_offset) < img_right.shape[0] and 0 <= (x - x_offset) < img_right.shape[1] else 0
            stitched_image[y, x] = (weight_matrix_left[y, x] * left_value + 
                                     weight_matrix_right[y, x] * right_value).astype(np.uint8)

    return stitched_image

# 拼接方式4：加權融合+模糊
def stitch_p2p_blending_blur(img_left, img_right, x_offset, y_offset):
    stitched_image = stitch_p2p_blending(img_left, img_right, x_offset, y_offset)

    # 漸層高斯模糊參數
    num_steps = 3  # 漸層的步數
    initial_blur = 9  # 初始模糊強度

    #處理左邊界的模糊
    if x_offset > 0:
        for step in range(num_steps):
            # 計算當前步驟的模糊內核大小，隨著步數減小
            blur_strength = initial_blur - step * (initial_blur // num_steps)
            if blur_strength % 2 == 0:
                blur_strength += 1  # 確保內核大小為奇數

            if step == 0:
                # 第一步：橫跨拼接線的模糊處理
                left_edge = max(0, x_offset - ((step + 1) * num_steps))
                right_edge = min(stitched_image.shape[1], x_offset + ((step + 1) * num_steps))
                
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
                right_end = min(stitched_image.shape[1], x_offset + ((step + 1) * num_steps))
                
                stitched_image[:, right_start:right_end] = cv2.GaussianBlur(
                    stitched_image[:, right_start:right_end], 
                    (blur_strength, blur_strength), 
                    0
                )
                
    # 處理右邊界的模糊
    for step in range(num_steps):
        # 計算當前步驟的模糊內核大小，隨著步數減小
        blur_strength = initial_blur - step * (initial_blur // num_steps)
        if blur_strength % 2 == 0:
            blur_strength += 1  # 確保內核大小為奇數

        if step == 0:
            # 第一步：橫跨拼接線的模糊處理
            left_edge = max(0, img_left.shape[1] - ((step + 1) * num_steps))
            right_edge = min(stitched_image.shape[1], img_left.shape[1] + ((step + 1) * num_steps))
            
            stitched_image[:, left_edge:right_edge] = cv2.GaussianBlur(
                stitched_image[:, left_edge:right_edge], 
                (blur_strength, blur_strength), 
                0
            )
        else:
            # 後續步驟：分別處理左右邊界
            # 處理左邊界
            left_start = max(0, img_left.shape[1] - (step + 1) * num_steps)
            left_end = img_left.shape[1] - step * num_steps
            
            stitched_image[:, left_start:left_end] = cv2.GaussianBlur(
                stitched_image[:, left_start:left_end], 
                (blur_strength, blur_strength), 
                0
            )

            # 處理右邊界
            right_start = img_left.shape[1] + ((step) * num_steps)
            right_end = min(stitched_image.shape[1], img_left.shape[1] + ((step + 1) * num_steps))
            
            stitched_image[:, right_start:right_end] = cv2.GaussianBlur(
                stitched_image[:, right_start:right_end], 
                (blur_strength, blur_strength), 
                0
            )

    return stitched_image

# 主程式：執行所有拼接方法
for left_image, right_image in zip(left_images, right_images):
    number = ''.join(filter(str.isdigit, left_image))
    image_path_left = os.path.join(data_folder, left_image)
    image_path_right = os.path.join(data_folder, right_image)
    
    img_left = cv2.imread(image_path_left, cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(image_path_right, cv2.IMREAD_GRAYSCALE)
    
    json_path_left = os.path.splitext(image_path_left)[0] + '.json'
    json_path_right = os.path.splitext(image_path_right)[0] + '.json'

    coord_left = get_coord_from_json(json_path_left, 'L5')
    coord_right = get_coord_from_json(json_path_right, 'L5')

    if coord_left is None or coord_right is None:
        raise ValueError("未能在JSON文件中找到指定的標籤 'L5'")

    x_offset = coord_left[0] - coord_right[0]
    y_offset = coord_left[1] - coord_right[1]

    # 保存結果
    os.makedirs(output_folder, exist_ok=True)

    result_p2p = stitch_p2p(img_left, img_right, x_offset, y_offset)
    cv2.imwrite(os.path.join(output_folder, f'p2p_{number}.png'), result_p2p)

    result_p2p_blur = stitch_p2p_blur(img_left, img_right, x_offset, y_offset)
    cv2.imwrite(os.path.join(output_folder, f'p2p_blur_{number}.png'), result_p2p_blur)

    result_p2p_blending = stitch_p2p_blending(img_left, img_right, x_offset, y_offset)
    cv2.imwrite(os.path.join(output_folder, f'p2p_blending_{number}.png'), result_p2p_blending)

    result_p2p_blending_blur = stitch_p2p_blending_blur(img_left, img_right, x_offset, y_offset)
    cv2.imwrite(os.path.join(output_folder, f'p2p_blending_blur_{number}.png'), result_p2p_blending_blur)

    print(f"Processed {left_image} and {right_image}")

print("All images processed.")
