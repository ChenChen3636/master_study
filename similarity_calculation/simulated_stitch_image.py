from PIL import Image
import json
import numpy as np
import cv2
import os
from math import radians, cos, sin

# 可以調參的變數，直接 clt+f 
# =============================#       
# 旋轉角度 : angle_of_rotation
# 高斯模糊 : add_gaussian_blur 
# =============================#



# 讀取標註的 L5 位置
with open('data\\stitched_evaluate_data\\image.json') as f:
    data = json.load(f)
    keypoint = data['shapes'][0]['points'][0]  # 假設 L5 是第一個標註

# 讀取影像
image = Image.open("data\\stitched_evaluate_data\\image.png")
image_width, image_height = image.size

# 剪裁寬度
cut_width = 800

# 將 image 轉為 numpy，cv2 要用
image_np = np.array(image)

# 剪裁影像
left_image = image_np[:, :cut_width]
right_image = image_np[:, -cut_width:]


# 計算新的 keypoint 在左右圖的座標位置
left_keypoint = keypoint if keypoint[0] < cut_width else None
right_keypoint = [keypoint[0] - (image_width - cut_width), keypoint[1]] if keypoint[0] >= (image_width - cut_width) else None 

# 添加高斯模糊
def add_gaussian_blur(image):
    return cv2.GaussianBlur(image, (11,11), 0)

# 旋轉影像
def rotate_image(image, angle, center):
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated, M

# 計算右圖變換後的 keypoint
def transform_keypoint(keypoint, matrix):
    x, y = keypoint
    point = np.array([x, y, 1])
    transformed_point = np.dot(matrix, point)
    return transformed_point[0], transformed_point[1]

# 加高斯
right_image_blur = add_gaussian_blur(right_image)
# 不要加高斯
# right_image_blur = right_image

# 設定旋轉角度和 center 位置
angle_of_rotation = 0 
center_of_rotation = (cut_width // 2, image_height)

right_image_rotated, rotation_matrix = rotate_image(right_image_blur, angle_of_rotation, center_of_rotation)

if right_keypoint:
    transformed_right_keypoint = transform_keypoint(right_keypoint, rotation_matrix)
else:
    transformed_right_keypoint = None
    
# 輸出結果
print("Left Image Keypoint:", left_keypoint)
print("Transformed Right Image Keypoint:", transformed_right_keypoint)

# Save the transformed right image
output_image_path = 'data\\stitched_evaluate_data\\transformed_right_image\gaussin_blur_7.png'
cv2.imwrite(output_image_path, right_image_rotated)

print(f"Transformed right image saved successfully to {output_image_path}.")

# 計算 x_offset 和 y_offset，準備進行拼接
x_offset = int(left_keypoint[0] - transformed_right_keypoint[0]) if left_keypoint and transformed_right_keypoint else 0
y_offset = int(left_keypoint[1] - transformed_right_keypoint[1]) if left_keypoint and transformed_right_keypoint else 0

#=================#
#   開始拼接 OAO 
#=================#

left_image = cv2.cvtColor(left_image, cv2.COLOR_RGB2GRAY)
right_image_rotated = cv2.cvtColor(right_image_rotated, cv2.COLOR_RGB2GRAY)

#拼接方式1：普通拼接
def stitch_p2p(left_image, img_right, x_offset, y_offset):
    height = max(left_image.shape[0], img_right.shape[0] + abs(y_offset))
    width = max(left_image.shape[1], img_right.shape[1] + abs(x_offset))
    stitched_image = np.zeros((height, width), dtype=np.uint8)
    stitched_image[:left_image.shape[0], :left_image.shape[1]] = left_image

    if x_offset >= 0 and y_offset >= 0:
        stitched_image[y_offset:y_offset + img_right.shape[0], x_offset:x_offset + img_right.shape[1]] = img_right
    elif x_offset >= 0 and y_offset < 0:
        stitched_image[:img_right.shape[0] + y_offset, x_offset:x_offset + img_right.shape[1]] = img_right[-y_offset:, :]

    return stitched_image

# 拼接方式2：普通拼接+模糊
def stitch_p2p_blur(left_image, img_right, x_offset, y_offset):
    stitched_image = stitch_p2p(left_image, right_image_rotated, x_offset, y_offset)

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
def stitch_p2p_blending(left_image, img_right, x_offset, y_offset):
    height = max(left_image.shape[0], img_right.shape[0] + abs(y_offset))
    width = max(left_image.shape[1], img_right.shape[1] + abs(x_offset))
    stitched_image = np.zeros((height, width), dtype=np.uint8)
    stitched_image[:left_image.shape[0], :left_image.shape[1]] = left_image

    weight_matrix_left = np.zeros((height, width), dtype=np.float32)
    weight_matrix_right = np.zeros((height, width), dtype=np.float32)

    overlap_start_x = max(x_offset, 0)
    overlap_end_x = min(left_image.shape[1], img_right.shape[1] + x_offset)

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
            left_value = left_image[y, x] if 0 <= y < left_image.shape[0] and 0 <= x < left_image.shape[1] else 0
            right_value = img_right[y - y_offset, x - x_offset] if 0 <= (y - y_offset) < img_right.shape[0] and 0 <= (x - x_offset) < img_right.shape[1] else 0
            stitched_image[y, x] = (weight_matrix_left[y, x] * left_value + 
                                     weight_matrix_right[y, x] * right_value).astype(np.uint8)

    return stitched_image

# 拼接方式4：加權融合+模糊
def stitch_p2p_blending_blur(left_image, img_right, x_offset, y_offset):
    stitched_image = stitch_p2p_blending(left_image, img_right, x_offset, y_offset)

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
            left_edge = max(0, left_image.shape[1] - ((step + 1) * num_steps))
            right_edge = min(stitched_image.shape[1], left_image.shape[1] + ((step + 1) * num_steps))
            
            stitched_image[:, left_edge:right_edge] = cv2.GaussianBlur(
                stitched_image[:, left_edge:right_edge], 
                (blur_strength, blur_strength), 
                0
            )
        else:
            # 後續步驟：分別處理左右邊界
            # 處理左邊界
            left_start = max(0, left_image.shape[1] - (step + 1) * num_steps)
            left_end = left_image.shape[1] - step * num_steps
            
            stitched_image[:, left_start:left_end] = cv2.GaussianBlur(
                stitched_image[:, left_start:left_end], 
                (blur_strength, blur_strength), 
                0
            )

            # 處理右邊界
            right_start = left_image.shape[1] + ((step) * num_steps)
            right_end = min(stitched_image.shape[1], left_image.shape[1] + ((step + 1) * num_steps))
            
            stitched_image[:, right_start:right_end] = cv2.GaussianBlur(
                stitched_image[:, right_start:right_end], 
                (blur_strength, blur_strength), 
                0
            )

    return stitched_image

# output 位置
output_folder = 'data\\stitched_evaluate_data\\Gaussin_blur'

# 創建輸出目錄
os.makedirs(output_folder, exist_ok=True)

# 進行四種拼接方法

# 拼接方式1：普通拼接
result_p2p = stitch_p2p(left_image, right_image_rotated, x_offset, y_offset)
cv2.imwrite(os.path.join(output_folder, 'R1.png'), result_p2p)

# 拼接方式2：普通拼接+模糊
result_p2p_blur = stitch_p2p_blur(left_image, right_image_rotated, x_offset, y_offset)
cv2.imwrite(os.path.join(output_folder, 'R2.png'), result_p2p_blur)

# 拼接方式3：加權融合
result_p2p_blending = stitch_p2p_blending(left_image, right_image_rotated, x_offset, y_offset)
cv2.imwrite(os.path.join(output_folder, 'R3.png'), result_p2p_blending)

# 拼接方式4：加權融合+模糊
result_p2p_blending_blur = stitch_p2p_blending_blur(left_image, right_image_rotated, x_offset, y_offset)
cv2.imwrite(os.path.join(output_folder, 'R4.png'), result_p2p_blending_blur)

# print(f"Processed {left_image} and {right_image_rotated}")