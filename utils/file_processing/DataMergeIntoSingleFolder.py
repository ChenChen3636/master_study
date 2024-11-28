import os
import shutil

def move_images(source_directory, target_directory):
    # 確保目標資料夾存在，如果不存在則創建它
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
        print(f"Created target directory: {target_directory}")
    
    # 定義圖片檔案的擴展名
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    
    # 遍歷source_directory及其所有子資料夾
    for root, dirs, files in os.walk(source_directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                file_path = os.path.join(root, file)
                target_path = os.path.join(target_directory, file)
                
                # 確保目標路徑的檔案名稱是唯一的
                count = 1
                original_target_path = target_path
                while os.path.exists(target_path):
                    name, extension = os.path.splitext(original_target_path)
                    target_path = f"{name}_{count}{extension}"
                    count += 1
                
                shutil.move(file_path, target_path)
                print(f"Moved: {file_path} -> {target_path}")

# 設定你的來源資料夾路徑
source_directory = 'view_all_images'
# 建立目標資料夾名稱為"view_all_images"
target_directory = 'all_images'

# 執行函式
move_images(source_directory, target_directory)
