import tensorflow as tf
import os
import glob
import numpy as np

def resize_and_pad_image(folder_path, target_size, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_files = glob.glob(os.path.join(folder_path, '*.[jp][pn]g')) # 遍歷資料夾中是 jpg, png 的檔案
    
    for image_file in image_files:
        image = tf.io.read_file(image_file)
        image = tf.image.decode_png(image, channels=3) # gray image channels is 1
        image = tf.image.convert_image_dtype(image, tf.float32) #將影像轉成tf看得懂的float32，也方便運用在歸一化
        
        resized_image = tf.image.resize_with_pad(
            image,
            target_size[0],
            target_size[1],
            method = tf.image.ResizeMethod.BILINEAR
        )
        
        # convert the processed image back to unit8
        resized_image_uint8 = tf.image.convert_image_dtype(resized_image, tf.uint8)
        
        #Encode the image as PNG and save
        output_path = os.path.join(output_folder, os.path.basename(image_file))
        resized_image_png = tf.io.encode_png(resized_image_uint8)
        tf.io.write_file(output_path, resized_image_png)
    
def resize_and_pad_label(folder_path, target_size, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    label_files = glob.glob(os.path.join(folder_path, '*.npy'))  # 遍历资料夹中的 NPY 文件
    
    for label_file in label_files:
        label = np.load(label_file)
        
        ### tf.image.reseze() 輸入預期至少 3D 陣列
        ### label是一個2D陣列 (height, width)，label[..., np.newaxis] 將label新增一個維度變成 (height,width,channel)
        
        label = label[..., np.newaxis]
        
        ### 使用'nearest' 使用最近鄰插法，將最近的像素值賦予目標像素，適用於圖像分割中的類別標籤
        resized_label = tf.image.resize_with_pad(
            label,
            target_size[0],
            target_size[1],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR  
        )
        
        ### 使用'squeeze' 移除大小為 1 的維度
        resized_label = tf.squeeze(resized_label).numpy()
        
        output_path = os.path.join(output_folder, os.path.basename(label_file))
        np.save(output_path, resized_label)
        
#image
image_dir = 'data\every_bone_similarity\LS_noL2'
image_output_dir = 'data\every_bone_similarity\\resize'

# label_dir = 'data\\voc_lumber\\voc_lumber_20240528\\SegmentationClass'
# label_output_dir = 'data\\voc_lumber\\voc_lumber_20240528\\SegmentationClass_Resize512'

target_size = (512,512)

resize_and_pad_image(image_dir,target_size,image_output_dir)
# resize_and_pad_label(label_dir, target_size, label_output_dir)
