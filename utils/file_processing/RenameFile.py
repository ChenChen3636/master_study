import os
from pathlib import Path



def simplify_filename(folder_path):
    
    for filepath in folder_path.iterdir():
        if filepath.is_file():
            parts = filepath.stem.split('_')
            new_filename = f"{parts[0]}_{parts[-1]}{filepath.suffix}"
            new_filepath = filepath.with_name(new_filename)
            print(f"Renaming {filepath} to {new_filepath}")
            os.rename(filepath, new_filepath)
            
            
from pathlib import Path

def rename_files_with_sequence(directory, prefix='', start=1):
    """
    重命名指定目录下的所有文件为流水号格式。
    
    :param directory: 要重命名文件的目录路径
    :param prefix: 新文件名的前缀（可选）
    :param start: 流水号开始的数字
    """
    # 确保路径是Path对象
    dir_path = Path(directory)
    counter = start
    
    # 按文件名排序，确保顺序
    sorted_files = sorted(dir_path.iterdir(), key=lambda x: x.name)
    
    for file_path in sorted_files:
        if file_path.is_file():  # 确保是文件
            # 生成新的文件名，保留原始扩展名
            new_filename = f"{prefix}{counter}{file_path.suffix}"
            new_file_path = file_path.with_name(new_filename)
            
            # 重命名文件
            file_path.rename(new_file_path)
            print(f"Renamed {file_path} to {new_file_path}")
            
            counter += 1  # 更新计数器

# 调用函数
directory_path = 'data\\temp'  # 替換路徑
rename_files_with_sequence(directory_path, prefix='lumber_', start=1)




