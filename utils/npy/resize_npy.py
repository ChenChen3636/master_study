import os
import numpy as np
import cv2

def resize_npy_files(input_folder, output_folder, target_size=(512, 512)):
    
    # 確保輸出資料夾存在，否則創建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 瀏覽資料夾中的所有.npy檔案
    for filename in os.listdir(input_folder):
        if filename.endswith(".npy"):
            file_path = os.path.join(input_folder, filename)
            
            # 讀取 .npy 檔案
            data = np.load(file_path)
            print(f"Processing {filename}, data shape: {data.shape}, data type: {data.dtype}")
            
            # 確保數據是2D或3D圖像，並檢查數據形狀
            if len(data.shape) == 2 or len(data.shape) == 3:
                # 如果數據不是 float32 或 uint8，轉換類型
                if data.dtype != np.uint8 and data.dtype != np.float32:
                    print(f"Converting {filename} data type from {data.dtype} to float32")
                    data = data.astype(np.float32)
                
                # 使用 cv2 進行 resize
                resized_data = cv2.resize(data, target_size, interpolation=cv2.INTER_AREA)
                
                # 儲存到新的 .npy 檔案中
                output_path = os.path.join(output_folder, filename)
                np.save(output_path, resized_data)
                print(f"Resized and saved: {output_path}")
            else:
                print(f"Skipping {filename}: unsupported shape {data.shape}")

# 使用方法
input_folder = "data\\voc_sacrum\\SegmentationClassNpy"  # 放置 .npy 檔案的資料夾路徑
output_folder = "data\\voc_sacrum\\SegmentationClassNpy_Resize512"  # 調整尺寸後儲存檔案的資料夾路徑
resize_npy_files(input_folder, output_folder)
