
import cv2
import numpy as np
import os

# 1. 平移 (Translation)
def apply_translation(image, x_translation, y_translation):
    rows, cols = image.shape
    M = np.float32([[1, 0, x_translation], [0, 1, y_translation]]) # type: ignore
    translated_image = cv2.warpAffine(image, M, (cols, rows)) # type: ignore
    return translated_image

# 2. 旋轉 (Rotation)
def apply_rotation(image, angle):
    rows, cols = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))
    return rotated_image

# 3. 縮放 (Scaling)
def apply_scaling(image, scale_x, scale_y):
    scaled_image = cv2.resize(image, None, fx=scale_x, fy=scale_y)
    return scaled_image

# 4. 剪切 (Shearing)
def apply_shearing(image, shear_factor):
    rows, cols = image.shape
    M = np.float32([[1, shear_factor, 0], [0, 1, 0]]) # type: ignore
    sheared_image = cv2.warpAffine(image, M, (cols, rows)) # type: ignore
    return sheared_image

# 5. 噪聲添加 (Noise Addition)
def apply_noise(image, noise_sigma):
    noise = np.random.normal(0, noise_sigma, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255)  # 確保像素值在有效範圍內
    return noisy_image.astype(np.uint8)

# 6. 仿射變換 (Affine Transformation)
def apply_affine_transform(image, delta_x, delta_y):
    rows, cols = image.shape
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]]) # type: ignore
    pts2 = np.float32([[50 + delta_x, 50 + delta_y], [200 + delta_x, 50 + delta_y], [50 + delta_x, 200 + delta_y]]) # type: ignore
    M = cv2.getAffineTransform(pts1, pts2) # type: ignore
    affine_image = cv2.warpAffine(image, M, (cols, rows))
    return affine_image

# 7. 非線性變形 (Non-linear Distortion)
def apply_non_linear_transform(image, frequency, amplitude):
    rows, cols = image.shape
    transformed_image = np.zeros_like(image)
    for i in range(rows):
        for j in range(cols):
            new_x = int(i + np.sin(j / frequency) * amplitude)
            new_y = j
            if 0 <= new_x < rows and 0 <= new_y < cols:
                transformed_image[i, j] = image[new_x, new_y]
    return transformed_image

# 主函式：應用所有變換並保存結果
def apply_transformations_and_save(image_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 應用並保存各種變換
    transformations = [
        ("translation", apply_translation(image, -30, 20)),
        ("rotation", apply_rotation(image, 15)),
        ("scaling", apply_scaling(image, 1.2, 1.2)),
        ("shearing", apply_shearing(image, 0.2)),
        ("noise", apply_noise(image, 5)),
        ("affine", apply_affine_transform(image, -20, 10)),
        ("non_linear", apply_non_linear_transform(image, 20, 10)),
    ]

    for name, transformed_image in transformations:
        output_path = os.path.join(output_folder, f"{name}.png")
        cv2.imwrite(output_path, transformed_image)
        print(f"Saved {name} transformed image to {output_path}")

# 範例使用
image_path = 'data\\stitched_evaluate_data\\S1.png'
output_folder = 'output\\transformed_images_3'

apply_transformations_and_save(image_path, output_folder)
