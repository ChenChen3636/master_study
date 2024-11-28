import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
import cv2

def calculate_mse(image1, image2):
    # 計算均方誤差 (MSE)
    mse = np.mean((image1 - image2) ** 2)
    return mse

def calculate_ssim(image1, image2):
    # 獲取影像最小邊長，確保窗口大小不超過影像區域
    min_side = min(image1.shape[0], image1.shape[1], image2.shape[0], image2.shape[1])
    
    # 確保窗口大小為奇數且小於等於最小邊長
    win_size = min(7, min_side) if min_side >= 7 else min_side - (min_side % 2 == 0)
    
    # 計算結構相似性指數 (SSIM)
    score, _ = ssim(image1, image2, full=True, win_size=win_size)
    return score

def calculate_pearson(image1, image2):
    # 計算Pearson相關係數
    image1_flat = image1.flatten()
    image2_flat = image2.flatten()
    pearson_corr, _ = pearsonr(image1_flat, image2_flat)
    return pearson_corr

def extract_overlap_region(image1, image2, img1_overlap_point, img2_overlap_point):
    # 圖片大小
    img_size = image1.shape[0]  
    
    # 計算平移量
    delta_x = img1_overlap_point[0] - img2_overlap_point[0]
    delta_y = img1_overlap_point[1] - img2_overlap_point[1]
    
    # 計算重疊區域的矩形座標
    # 左上角
    overlap_x1 = max(0, delta_x)
    overlap_y1 = max(0, delta_y)
    
    # 右下角
    overlap_x2 = min(img_size, img_size + delta_x)
    overlap_y2 = min(img_size, img_size + delta_y)
    
    # 計算右圖的重疊矩形座標
    right_overlap_x1 = overlap_x1 - delta_x
    right_overlap_y1 = overlap_y1 - delta_y
    right_overlap_x2 = overlap_x2 - delta_x
    right_overlap_y2 = overlap_y2 - delta_y
    
    # 提取重疊區域
    overlap_region1 = image1[overlap_y1:overlap_y2, overlap_x1:overlap_x2]
    overlap_region2 = image2[right_overlap_y1:right_overlap_y2, right_overlap_x1:right_overlap_x2]
    
    return overlap_region1, overlap_region2

def calculate_similarity(image1, image2, img1_overlap_point, img2_overlap_point):
    # 提取重疊區域
    overlap_region1, overlap_region2 = extract_overlap_region(image1, image2, img1_overlap_point, img2_overlap_point)

    # 確保重疊區域不是空的
    if overlap_region1.size == 0 or overlap_region2.size == 0:
        raise ValueError("重疊區域大小為0，請檢查座標是否正確")

    # 計算相似度
    mse_score = calculate_mse(overlap_region1, overlap_region2)
    ssim_score = calculate_ssim(overlap_region1, overlap_region2)
    pearson_score = calculate_pearson(overlap_region1, overlap_region2)

    return mse_score, ssim_score, pearson_score


# 示例用法
image1_path = 'data\\data_calculate_similarity_different_point\\lumber_3.jpg'  # 假設為隨機產生的影像
image2_path = 'data\\data_calculate_similarity_different_point\\sacrum_3.jpg'  # 假設為隨機產生的影像

image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

img1_point = (335,243)
img2_point = (110,208)

mse_score, ssim_score, pearson_score = calculate_similarity(image1, image2, img1_point, img2_point)

print(f"MSE: {mse_score}")
print(f"SSIM: {ssim_score}")
print(f"Pearson Correlation: {pearson_score}")
