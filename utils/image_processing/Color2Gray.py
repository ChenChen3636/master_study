from PIL import Image
import os

target_dir = 'data\S1_data\SegmentationClass'

for fname in os.listdir(target_dir):
    if fname.endswith('png') and not fname.startswith('.'):
        
        file_path = os.path.join(target_dir, fname)
        img = Image.open(file_path)
        #轉換圖像為灰階， "L" 模式代表灰階(Grayscale)
        img = img.convert("L")
        #將所有非黑色的像素都轉成白色(255)
        pixels = img.load()
        for i in range(img.width):
            for j in range(img.height):
                if pixels[i,j] > 0:
                    pixels[i,j] = 255
                    
        img.save(file_path)
        
print("finished!")