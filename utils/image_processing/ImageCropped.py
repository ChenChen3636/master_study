import os
from PIL import Image

source_directory = 'data\every_bone_similarity\LS_noL2'
target_directory = 'data\every_bone_similarity\LS_noL2_cropped'

if not os.path.exists(target_directory):
    os.makedirs(target_directory)
    
    
for filename in os.listdir(source_directory):
    if filename.lower().endswith('png'):
        with Image.open(os.path.join(source_directory, filename)) as img:
            
            x,y = 0,54 #剪裁座標
            width = 512 
            height = 404

            right = x + width
            bottom = y + height


            cropped_img = img.crop((x, y, right, bottom))
                        
            output_path = os.path.join(target_directory, filename)
            cropped_img.save(output_path)
            
print("cropped ultrasound images finish")
