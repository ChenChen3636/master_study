import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os

# 定義影像增強函數
def image_aug(image_path, output_path, contrast_factor, brightness_factor):
    # 讀取影像
    image = Image.open(image_path)

    # 對比度增強
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)

    # 亮度增強
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)

    # 儲存增強後的影像
    image.save(output_path)

# 批次處理影像
def batch_process_images(input_folder, output_folder, contrast_factor, brightness_factor):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            image_aug(input_path, output_path, contrast_factor, brightness_factor)
            print(f"Processed {filename}")

# 使用範例
input_folder = 'data\\voc_0711_lumber\\JPEGImages'
output_folder = 'data\\voc_0711_lumber\\aug_image'

contrast_factor = 3.0
brightness_factor = 1.0
batch_process_images(input_folder, output_folder,contrast_factor, brightness_factor)
