import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim

def calculate_mse(image1, image2):
    # 計算均方誤差 (MSE)
    mse = np.mean((image1 - image2) ** 2)
    return round(mse, 4)

def calculate_ssim(image1, image2):
    # 計算結構相似性指數 (SSIM)
    score, _ = ssim(image1, image2, full=True)
    return round(score, 4)

def calculate_psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 10 * np.log10(max_pixel**2 / mse)
    return round(psnr, 4)

def compare_with_folder(original_image_path, folder_path, cut_width):
    # 讀取原圖
    original_img = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

    # 獲取原圖尺寸
    target_height, target_width = original_img.shape

    # 確保 cut_width 合法
    if cut_width > target_width:
        raise ValueError(f"cut_width ({cut_width}) 不能大於影像寬度 ({target_width})。")

    # 計算重疊區域的寬度
    overlap_width = 2 * cut_width - target_width
    if overlap_width <= 0:
        # 無重疊區域
        print("給定的 cut_width 沒有重疊區域。")
        overlap_slice = None
        non_overlap_left_slice = slice(0, cut_width)
        non_overlap_right_slice = slice(target_width - cut_width, target_width)
    else:
        # 存在重疊區域
        overlap_start = target_width - cut_width
        overlap_end = cut_width - 1

        overlap_slice = slice(overlap_start, overlap_end + 1)  # +1 因為切片結束索引是獨占的

        # 非重疊區域
        non_overlap_left_slice = slice(0, overlap_start)
        non_overlap_right_slice = slice(overlap_end + 1, target_width)

    # 初始化結果字典
    results_whole = {}
    results_overlap = {}
    results_non_overlap = {}

    for file_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, file_name)

        # 確保文件是圖片
        if os.path.isfile(image_path) and file_name.endswith('.png'):
            # 讀取資料夾中的圖片
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # 確保圖片尺寸一致
            if img.shape != original_img.shape:
                raise ValueError(f"影像 {file_name} 的尺寸與原始影像不同。")

            # 計算整張圖片的 MSE、SSIM 和 PSNR
            mse_whole = calculate_mse(original_img, img)
            ssim_whole = calculate_ssim(original_img, img)
            psnr_whole = calculate_psnr(original_img, img)
            results_whole[file_name.replace('.png', '')] = (mse_whole, ssim_whole, psnr_whole)

            # 檢查是否存在重疊區域
            if overlap_slice is None:
                # 無重疊區域
                mse_overlap = None
                ssim_overlap = None
                psnr_overlap = None
                results_overlap[file_name.replace('.png', '')] = (mse_overlap, ssim_overlap, psnr_overlap)

                # 非重疊區域是左右兩側的 cut_width 區域
                original_non_overlap = np.hstack((
                    original_img[:, non_overlap_left_slice],
                    original_img[:, non_overlap_right_slice]
                ))
                img_non_overlap = np.hstack((
                    img[:, non_overlap_left_slice],
                    img[:, non_overlap_right_slice]
                ))

                # 計算非重疊區域的 MSE、SSIM 和 PSNR
                mse_non_overlap = calculate_mse(original_non_overlap, img_non_overlap)
                ssim_non_overlap = calculate_ssim(original_non_overlap, img_non_overlap)
                psnr_non_overlap = calculate_psnr(original_non_overlap, img_non_overlap)
                results_non_overlap[file_name.replace('.png', '')] = (mse_non_overlap, ssim_non_overlap, psnr_non_overlap)
            else:
                # 提取重疊區域
                original_overlap = original_img[:, overlap_slice]
                img_overlap = img[:, overlap_slice]

                # 計算重疊區域的 MSE、SSIM 和 PSNR
                mse_overlap = calculate_mse(original_overlap, img_overlap)
                ssim_overlap = calculate_ssim(original_overlap, img_overlap)
                psnr_overlap = calculate_psnr(original_overlap, img_overlap)
                results_overlap[file_name.replace('.png', '')] = (mse_overlap, ssim_overlap, psnr_overlap)

                # 提取非重疊區域並拼接
                original_non_overlap = np.hstack((
                    original_img[:, non_overlap_left_slice],
                    original_img[:, non_overlap_right_slice]
                ))
                img_non_overlap = np.hstack((
                    img[:, non_overlap_left_slice],
                    img[:, non_overlap_right_slice]
                ))

                # 計算非重疊區域的 MSE、SSIM 和 PSNR
                mse_non_overlap = calculate_mse(original_non_overlap, img_non_overlap)
                ssim_non_overlap = calculate_ssim(original_non_overlap, img_non_overlap)
                psnr_non_overlap = calculate_psnr(original_non_overlap, img_non_overlap)
                results_non_overlap[file_name.replace('.png', '')] = (mse_non_overlap, ssim_non_overlap, psnr_non_overlap)

    return results_whole, results_overlap, results_non_overlap


original_image_path = 'data\\stitched_evaluate_data\image.png'
folder_path = 'data\stitched_evaluate_data\output_results'

# 指定 cut_width
cut_width = 800

results_whole, results_overlap, results_non_overlap = compare_with_folder(original_image_path, folder_path, cut_width)

# 輸出結果
print("整張影像：")
for file_name, (mse, ssim_value, psnr_value) in results_whole.items():
    print(f"  {file_name} : MSE = {mse}, SSIM = {ssim_value}, PSNR = {psnr_value}")

print("\n重疊區域：")
for file_name, (mse, ssim_value, psnr_value) in results_overlap.items():
    if mse is not None:
        print(f"  {file_name} : MSE = {mse}, SSIM = {ssim_value}, PSNR = {psnr_value}")
    else:
        print(f"  {file_name} : 無重疊區域")

print("\n非重疊區域：")
for file_name, (mse, ssim_value, psnr_value) in results_non_overlap.items():
    print(f"  {file_name} : MSE = {mse}, SSIM = {ssim_value}, PSNR = {psnr_value}")
