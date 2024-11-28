import cv2
import numpy as np

from matplotlib import pyplot as plt

def add_noise(img, n):
    img2 = img
    for _ in range(n):
        x = int(np.random.random() * img.shape[0])
        y = int(np.random.random() * img.shape[1])
        img2[x, y] = 255 # 白色的灰階值是 255
    return img2

def cropped_img(img):
    
    x,y = 167,59 #剪裁座標
    width = 690 
    height = 551

    cropped_img = img[y:y+height, x:x+width]

    return cropped_img

if __name__ == '__main__':
    
    # 輸入原始影像
    # img = cv2.imread('./input/turing.jpg',0)

    # 增加白噪
    # noise_img = add_noise(img, 40000)
    # cv2.imwrite('./input/white_noise.jpg', noise_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

    img = cv2.imread('ultrasound/8/b.png',0)
    
    img = cropped_img(img)

    # 將原圖一起輸出，較容易比較結果
    plt.figure(figsize=(10,20))

    # 用 counter 將每張圖依序畫在 plt 上
    counter = 1
    num_value = len(range(3,6,2))
    nrows = num_value * 3
  

    # 設定一序列的 mask ，並逐一作用在原始圖像上
    for mask_size in range(11, 14, 2):
        
        plt.subplot(nrows,3,counter)
        plt.imshow(img, cmap= 'gray')
        plt.title('original')
        plt.xticks([])
        plt.yticks([])
        counter += 1
        
        # 採用中值濾波器
        median_image = cv2.medianBlur(img, mask_size)
        # 將每種 mask size 都先畫到plt 最後一次輸出比較
        #plt.subplot(nrows,ncols,index)
        plt.subplot(nrows,3,counter)
        plt.imshow(median_image, cmap = 'gray')
        plt.title('Median Blur Image, mask size: '+str(mask_size))
        plt.xticks([])
        plt.yticks([])
        counter += 1
        
        smoothing_mask = np.ones((mask_size, mask_size), np.float32)/ (mask_size ** 2)
        # 將 mask apply 到原始圖像做 Smoothing
        smoothing_img = cv2.filter2D(img,-1,smoothing_mask)
        plt.subplot(nrows,3,counter)
        plt.imshow(smoothing_img, cmap = 'gray')
        plt.title('Smoothing Image, mask size: '+str(mask_size))
        plt.xticks([])
        plt.yticks([])
        counter += 1
        
        print(counter)

    # 儲存所有輸出的圖像
    plt.savefig('./output/Comparison_Result.png', bbox_inches='tight', dpi=400)