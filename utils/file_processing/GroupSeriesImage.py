import os
from shutil import move

def classify_and_move_files(source_dir):
    def ensure_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
            
    prefix_to_folder = {}
    
    for filename in os.listdir(source_dir):
        
        if filename.endswith('.png'):
            prefix = filename.split('_')[0] #取得檔名開頭部分
            
            if prefix not in prefix_to_folder:
                folder_name = prefix
                target_dir = os.path.join(source_dir, folder_name)
                ensure_dir(target_dir)
                prefix_to_folder[prefix] = target_dir
                
            source_file = os.path.join(source_dir, filename)
            target_file = os.path.join(prefix_to_folder[prefix], filename)
            move(source_file, target_file)
            print(f'Moved: {source_file} -> {target_file}')
            
source_directory = './view_all_images' 
classify_and_move_files(source_directory)
                
                