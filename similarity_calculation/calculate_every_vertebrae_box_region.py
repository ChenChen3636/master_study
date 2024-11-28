import cv2
import numpy as np
import json
import os
import glob
import csv
from skimage.metrics import structural_similarity as ssim

def calculate_mse(image1, image2):
    # 計算均方誤差（MSE）
    mse = np.mean((image1.astype("float") - image2.astype("float")) ** 2)
    return round(mse, 4)

def calculate_ssim(image1, image2):
    # 計算結構相似性指數（SSIM）
    score, _ = ssim(image1, image2, full=True)
    return round(score, 4)

def read_annotations(json_path):
    # 讀取 JSON 檔案並返回一個字典，包含椎骨名稱及其對應的座標
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    
    vertebrae = {}
    shapes = annotations.get('shapes', [])
    for shape in shapes:
        label = shape.get('label')
        points = shape.get('points', [])
        if points and len(points) > 0:
            # 假設每個標註只包含一個點
            x, y = points[0]
            vertebrae[label] = [x, y]
    return vertebrae

def read_l5_box(json_path):
    # 讀取 Sacrum JSON，找到 L5_box 的 rectangle 範圍
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    
    for shape in annotations.get('shapes', []):
        if shape.get('label') == 'L5_box' and shape.get('shape_type') == 'rectangle':
            # 假設 points 只包含兩個點，分別是左上角和右下角的座標
            points = shape.get('points', [])
            if len(points) == 2:
                x1, y1 = points[0]  # 左上角座標
                x2, y2 = points[1]  # 右下角座標
                return int(x1), int(y1), int(x2), int(y2)
    
    return None  # 如果找不到 L5_box 則返回 None

def stitch_p2p(left_image, img_right, x_offset, y_offset):
    # 計算拼接後影像的尺寸
    height = max(left_image.shape[0], img_right.shape[0] + abs(y_offset))
    width = max(left_image.shape[1], img_right.shape[1] + abs(x_offset))
    stitched_image = np.zeros((height, width), dtype=np.uint8)

    # 將左邊的影像放到拼接圖像上
    stitched_image[:left_image.shape[0], :left_image.shape[1]] = left_image

    # 計算右邊影像在拼接圖像中的位置
    if x_offset >= 0 and y_offset >= 0:
        stitched_image[y_offset:y_offset + img_right.shape[0], x_offset:x_offset + img_right.shape[1]] = img_right
    elif x_offset >= 0 and y_offset < 0:
        stitched_image[0:img_right.shape[0] + y_offset, x_offset:x_offset + img_right.shape[1]] = img_right[-y_offset:, :]
    elif x_offset < 0 and y_offset >= 0:
        stitched_image[y_offset:y_offset + img_right.shape[0], 0:img_right.shape[1] + x_offset] = img_right[:, -x_offset:]
    else:
        stitched_image[0:img_right.shape[0] + y_offset, 0:img_right.shape[1] + x_offset] = img_right[-y_offset:, -x_offset:]

    return stitched_image

def compute_overlap_region(left_image, right_image, x_offset, y_offset, l5_box):
    # 提取 L5_box 區域的座標
    x1_box, y1_box, x2_box, y2_box = l5_box
    
    # 確保 L5_box 座標正確排序（左上角 < 右下角）
    x1_box, x2_box = min(x1_box, x2_box), max(x1_box, x2_box)
    y1_box, y2_box = min(y1_box, y2_box), max(y1_box, y2_box)
    
    # 打印 L5_box 座標和偏移量
    # print(f"L5_box 座標: ({x1_box}, {y1_box}), ({x2_box}, {y2_box})")
    # print(f"x_offset: {x_offset}, y_offset: {y_offset}")

    # 計算重疊區域的座標範圍
    x1_left = max(0, x1_box + x_offset)
    y1_left = max(0, y1_box + y_offset)
    x2_left = min(left_image.shape[1], x2_box + x_offset)
    y2_left = min(left_image.shape[0], y2_box + y_offset)

    x1_right = max(-x_offset, x1_box)
    y1_right = max(-y_offset, y1_box)
    x2_right = x1_right + (x2_left - x1_left)
    y2_right = y1_right + (y2_left - y1_left)
    
    # 打印計算後的重疊區域座標
    # print(f"Left Image Overlap: x1_left={x1_left}, y1_left={y1_left}, x2_left={x2_left}, y2_left={y2_left}")
    # print(f"Right Image Overlap: x1_right={x1_right}, y1_right={y1_right}, x2_right={x2_right}, y2_right={y2_right}")

    # 檢查是否存在重疊區域並且是否超出邊界
    if x1_left >= x2_left or y1_left >= y2_left or x1_right >= right_image.shape[1] or y1_right >= right_image.shape[0]:
        print(f"重疊區域超過邊界，跳過本次計算。")
        return None, None
    
    # 檢查是否右圖的重疊區域超過邊界
    if x2_right > right_image.shape[1] or y2_right > right_image.shape[0]:
        print(f"右圖重疊區域超出邊界，跳過本次計算。")
        return None, None

    # 提取左圖和右圖的重疊區域
    left_overlap = left_image[y1_left:y2_left, x1_left:x2_left]
    right_overlap = right_image[y1_right:y2_right, x1_right:x2_right]

    return left_overlap, right_overlap

def main():
    # 設定 Lumbar 和 Sacrum 影像目錄的路徑
    lumbar_dir = 'data\\every_bone_similarity\\lumbar'  
    sacrum_dir = 'data\\every_bone_similarity\\sacrum'  

    # 獲取 Lumbar 影像列表
    lumbar_image_paths = glob.glob(os.path.join(lumbar_dir, '*.png')) 

    # 創建保存拼接結果的目錄
    stitched_output_dir = 'output\every_bone_similarity\\box_region\\stitched_results'
    os.makedirs(stitched_output_dir, exist_ok=True)

    # 創建保存重疊區域結果的目錄
    overlap_output_dir = 'output\every_bone_similarity\\box_region\\overlap_results'
    os.makedirs(overlap_output_dir, exist_ok=True)

    # 初始化結果列表
    all_results = []
    correct_SSIM_matches = 0  # 計算正確匹配到 L5 的次數
    correct_MSE_matches = 0
    correct_combined_matches = 0
    total_images = 0  # 總共匹配的影像數量
    # 定義加權參數 alpha 和 beta
    alpha = 0.8  # MSE 的權重
    beta = 0.2   # SSIM 的權重

    # 遍歷每個 Lumbar 影像
    for lumbar_image_path in lumbar_image_paths:
        # 獲取檔名以找到對應的 Sacrum 影像
        basename = os.path.basename(lumbar_image_path)
        name_no_ext = os.path.splitext(basename)[0]

        # 根據 Lumbar 的檔案名稱生成對應的 Sacrum 檔案名稱
        sacrum_name_no_ext = name_no_ext.replace('L', 'S')
        sacrum_image_path = os.path.join(sacrum_dir, sacrum_name_no_ext + '.png')  
        lumbar_annotation_path = os.path.join(lumbar_dir, name_no_ext + '.json')
        sacrum_annotation_path = os.path.join(sacrum_dir, sacrum_name_no_ext + '.json')

        # 檢查是否存在對應的 Sacrum 影像
        if not os.path.exists(sacrum_image_path):
            print(f"未找到對應的 Sacrum 影像: {basename}，跳過該影像。")
            continue

        # 讀取影像
        left_image = cv2.imread(lumbar_image_path, cv2.IMREAD_GRAYSCALE)
        right_image = cv2.imread(sacrum_image_path, cv2.IMREAD_GRAYSCALE)

        # 讀取標註
        left_vertebrae = read_annotations(lumbar_annotation_path)
        right_vertebrae = read_annotations(sacrum_annotation_path)

        # 確保 Sacrum 標註中包含 'L5'
        if 'L5' not in right_vertebrae:
            print(f"Sacrum 標註中未找到 'L5'，跳過該影像: {basename}。")
            continue

        # 讀取 L5_box 區域
        l5_box = read_l5_box(sacrum_annotation_path)
        if l5_box is None:
            print(f"Sacrum 標註中未找到 'L5_box'，跳過該影像: {basename}。")
            continue

        right_L5_coord = right_vertebrae['L5']

        # 初始化當前影像的結果
        image_results = []

        # 遍歷 Lumbar 影像中的每個椎骨
    
    
        for vertebra_name, left_coord in left_vertebrae.items():
            # 計算位移偏移量
            x_offset = int(left_coord[0] - right_L5_coord[0])
            y_offset = int(left_coord[1] - right_L5_coord[1])
            
            # print(f"對齊 Lumbar 椎骨 {vertebra_name} 到 Sacrum L5: x_offset = {x_offset}, y_offset = {y_offset}")


            # 拼接影像
            stitched_image = stitch_p2p(left_image, right_image, x_offset, y_offset)

            # 計算重疊區域
            left_overlap, right_overlap = compute_overlap_region(left_image, right_image, x_offset, y_offset, l5_box)

            # 檢查重疊區域是否存在
            if left_overlap is None or right_overlap is None or left_overlap.size == 0 or right_overlap.size == 0:
                print(f"在將 {vertebra_name} 對齊 Sacrum 的 L5 後無重疊區域: {basename}。")
                continue

            # 計算相似度指標
            mse_overlap = calculate_mse(left_overlap, right_overlap)
            ssim_overlap = calculate_ssim(left_overlap, right_overlap)

            # 保存拼接後的影像
            stitched_output_path = os.path.join(stitched_output_dir, f'{name_no_ext}_stitched_{vertebra_name}.png')
            cv2.imwrite(stitched_output_path, stitched_image)

            # 保存重疊區域影像
            overlap_combined = np.hstack((left_overlap, right_overlap))
            overlap_output_path = os.path.join(overlap_output_dir, f'{name_no_ext}_overlap_{vertebra_name}.png')
            cv2.imwrite(overlap_output_path, overlap_combined)

            # 保存結果
            image_results.append({
                'Lumbar_Image': basename,
                'Vertebra': vertebra_name,
                'MSE': mse_overlap,
                'SSIM': ssim_overlap
            })

        # 找到 SSIM 值最高和MSE最小的椎骨（最相似）
        if image_results:
            best_match_SSIM = max(image_results, key=lambda x: x['SSIM'])
            best_match_MSE = min(image_results, key=lambda x: x['MSE'])
            
            print(f"Lumbar 影像 {basename} 中與 Sacrum 的 L5 最相似的椎骨是 {best_match_SSIM['Vertebra']}，SSIM = {best_match_SSIM['SSIM']}")
            print(f"Lumbar 影像 {basename} 中與 Sacrum 的 L5 最相似的椎骨（MSE）是 {best_match_MSE['Vertebra']}，MSE = {best_match_MSE['MSE']}")


            # 標記最相似的椎骨
            for result in image_results:
                result['Most_Similar_SSIM'] = result['Vertebra'] == best_match_SSIM['Vertebra']
                result['Most_Similar_MSE'] = result['Vertebra'] == best_match_MSE['Vertebra']
                combined_score = alpha * (1 / (1 + result['MSE']))**2 + beta * result['SSIM']                result['Combined_Score'] = combined_score
                
            # 找到組合分數最高的椎骨
            best_combined_match = max(image_results, key=lambda x: x['Combined_Score'])

            print(f"Lumbar 影像 {basename} 中與 Sacrum 的 L5 最相似的椎骨（組合分數）是 {best_combined_match['Vertebra']}，組合分數 = {best_combined_match['Combined_Score']}")


            # 檢查是否最相似的椎骨是 L5
            if best_match_SSIM['Vertebra'] == 'L5':
                correct_SSIM_matches += 1  # 正確匹配到 L5
                
            # 檢查是否最相似的椎骨是 L5（對於 MSE）
            if best_match_MSE['Vertebra'] == 'L5':
                correct_MSE_matches += 1  # 正確匹配到 L5（MSE）
                
            # 判斷最相似的椎骨是否是 L5
            if best_combined_match['Vertebra'] == 'L5':
                correct_combined_matches += 1  # 正確匹配到 L5（組合分數）

            total_images += 1  # 處理的影像數量

        else:
            print(f"{basename} 沒有有效的比較結果。")
            continue

        # 將當前影像的所有結果加入主結果列表
        all_results.extend(image_results)
        
    # 計算準確率
    if total_images > 0:
        SSIM_accuracy = (correct_SSIM_matches / total_images) * 100
        MSE_accuracy = (correct_MSE_matches / total_images) * 100
        combined_accuracy = (correct_combined_matches / total_images) * 100
        print(f"SSIM 準確率：{SSIM_accuracy:.2f}%")
        print(f"MSE 準確率：{MSE_accuracy:.2f}%")
        print(f"組合匹配準確率：{combined_accuracy:.2f}%")
    else:
        print("沒有可用的影像進行匹配。")

    # 保存結果到 CSV 檔案
    csv_file = 'output\every_bone_similarity\\box_region\\similarity_results_v2.csv'
    with open(csv_file, 'w', newline='') as csvfile:
        fieldnames = ['Lumbar_Image', 'Vertebra', 'MSE', 'SSIM', 'Most_Similar_SSIM', 'Most_Similar_MSE', 'Combined_Score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    print(f"結果已保存到 {csv_file}")
    
if __name__ == "__main__":
    main()
